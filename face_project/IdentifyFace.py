import face_recognition
from face_project import train_face




def identiyPerson(image_array):
    # face_location = face_recognition.face_locations(image_array, number_of_times_to_upsample=1)
    unknown_face_encodings = face_recognition.face_encodings(image_array)
    if len(unknown_face_encodings) > 0:
        unknown_face = unknown_face_encodings[0]
        train_facemap = train_face.getface_encoding_map()
        for name, face_encodings in train_facemap.items():
            print(name)
            for face_encoding in face_encodings:
                result = face_recognition.compare_faces(face_encoding, unknown_face, tolerance=0.6)
                if result[0]:
                    return name

    return "Unknown"


