from __future__ import annotations
import os
import cv2
import sys
import json
import boto3
import shutil
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import PIL.Image


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def normalize_coords(x1, y1, w, h, image_w, image_h):
    return ["%.6f" % ((2*x1 + w)/(2*image_w)), "%.6f" % ((2*y1 + h)/(2*image_h)), "%.6f" % (w/image_w), "%.6f" % (h/image_h)]


def convert_annotations(manifest_jsonl, folder, dict_key, label_name):

    for manifest_jsonl_line in manifest_jsonl:
        image_uri = manifest_jsonl_line['source-ref']
        filename = os.path.basename(image_uri)
        # template['annotation']['filename'] = filename
        ext, filename = filename[::-1].split('.', 1)
        ext = ext[::-1]
        filename = filename[::-1]
        image_path = f"custom_dataset/{folder}/{filename}.{ext}"
        xml_path = f"custom_dataset/{folder}/{filename}.xml"
        bucket, key = image_uri[5:].split('/', 1)
        s3.Bucket(bucket).download_file(key, image_path)
        image = PIL.Image.open(image_path)
        if image.format != 'JPEG' and image.format.lower() != 'png':
            os.remove(image_path)
            continue
        # img_file = cv2.imread(image_path)
        annotation = Element('annotation')
        src = SubElement(annotation, 'source')
        SubElement(src, 'database').text = 'Unknown'
        SubElement(annotation, 'path').text = image_path
        SubElement(annotation, 'filename').text = f'{filename}.{ext}'
        SubElement(annotation, 'folder').text = f"custom_dataset/{folder}"
        SubElement(annotation, 'segmented').text = "0"
        size = SubElement(annotation, 'size')
        SubElement(size, 'height').text = str(
            manifest_jsonl_line[dict_key]['image_size'][0]['height'])
        SubElement(size, 'width').text = str(
            manifest_jsonl_line[dict_key]['image_size'][0]['width'])
        SubElement(size, 'depth').text = str(
            manifest_jsonl_line[dict_key]['image_size'][0]['depth'])
        for annotation_data in manifest_jsonl_line[dict_key]['annotations']:
            bbx_object = SubElement(annotation, 'object')
            SubElement(bbx_object, 'name').text = label_name
            SubElement(bbx_object, 'pose').text = 'Unspecified'
            SubElement(bbx_object, 'truncated').text = "0"
            SubElement(bbx_object, 'difficult').text = "0"
            bndbox = SubElement(bbx_object, 'bndbox')
            x = annotation_data['left']
            y = annotation_data['top']
            w = annotation_data['width'] + x
            h = annotation_data['height'] + y
            SubElement(bndbox, 'xmin').text = str(x)
            SubElement(bndbox, 'ymin').text = str(y)
            SubElement(bndbox, 'xmax').text = str(w)
            SubElement(bndbox, 'ymax').text = str(h)

            # img_file = cv2.rectangle(img_file, (x, y), (w, h), (255, 0, 0), 2)

        with open(xml_path, 'w') as f:
            f.write(prettify(annotation))
        # cv2.imwrite(f'annotation/{filename}.{ext}', img_file)


s3 = boto3.resource('s3')
smgt_client = boto3.client('sagemaker', region_name='us-west-2')
queue_name = sys.argv[1]  # -- "V3-Cup-BatchP-BBox-Inhouse-chain"
label = sys.argv[2]  # -- "cup"
train_samples = int(sys.argv[3])  # -- 25
test_val = int(train_samples * 0.10)
manifest_jsonl = []


# refresh folders
if os.path.exists('custom_dataset'):
    shutil.rmtree('custom_dataset')
if os.path.exists('anntoation'):
    shutil.rmtree('anntoation')
cwd = os.getcwd()

train_fodler = cwd + "/custom_dataset/train"
validate_fodler = cwd + "/custom_dataset/validate"
Path(train_fodler).mkdir(parents=True, exist_ok=True)
Path(validate_fodler).mkdir(parents=True, exist_ok=True)
Path('annotation').mkdir(parents=True, exist_ok=True)

response = smgt_client.describe_labeling_job(
    LabelingJobName=queue_name
)


if response["LabelingJobStatus"] == "Completed":
    output_manifest_path = response["OutputConfig"]["S3OutputPath"] + \
        f"{queue_name}/manifests/output/output.manifest"
    bucket, key = output_manifest_path[5:].split('/', 1)
    manifest_obj = s3.Object(bucket, key)
    manifest_lines = manifest_obj.get()['Body'].read().decode().splitlines()
    for manifest_line in manifest_lines:
        manifest_data = json.loads(manifest_line)
        manifest_jsonl.append(manifest_data)
    print(f'Finished parsing manifest: {queue_name}')
else:
    print('Job not completed yet')
    sys.exit()

key = [keys for keys in manifest_jsonl[0].keys() if not keys.endswith(
    '-metadata') and not keys == 'source-ref'][0]
print(key)

image_count = len(manifest_jsonl)

training_set = manifest_jsonl[:train_samples]
convert_annotations(training_set, "train", key, label)
val_set = manifest_jsonl[train_samples:train_samples + test_val]
convert_annotations(val_set, "validate", key, label)
