import face_recognition

face_encodingmap = {};


def trainface(image, name):
    face_location = face_recognition.face_locations(image, number_of_times_to_upsample=2)
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_location)
    addtomap(face_encoding, name)


def addtomap(face_encoding, name):
    if name in face_encodingmap:
        face_encodingmap[name].append(face_encoding)
    else:
        face_encodingmap[name] = [face_encoding]


def getface_encoding_map():
    return face_encodingmap


def train_inital():
    image = face_recognition.load_image_file("itsme.jpeg")
    trainface(image, "ASK")

train_inital()
