import json

# new_filenames = {}

a_file = open("dataset/val/annotations.json", "r")
json_object = json.load(a_file)
for img in json_object['images']:
    new_filename = img['file_name'].split("/")[-1]
    img['file_name'] = new_filename
    # new_filenames[int(img['id'])] = new_filename
a_file.close()

print(json_object)

a_file = open("dataset/val/annotations.json", "w")
json.dump(json_object, a_file)
a_file.close()