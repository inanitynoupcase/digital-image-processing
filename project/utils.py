import face_recognition

import numpy as np



def haar_face_to_points(face):
    x1, y1, w, h = face
    x2 = x1 + w 
    y2 = y1 + h

    return x1, y1, x2, y2


def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2


def dnn_detection_to_points(detection, width, height):
    x1 = int(detection[3] * width)
    y1 = int(detection[4] * height)
    x2 = int(detection[5] * width)
    y2 = int(detection[6] * height)

    return x1, y1, x2, y2



def load_image(image_path):
    """ Helper function to load image, 
    Parameters
    ----------
    image_path : str
        Image path
    Returns
    -------
    numpy.darray
        RGB image stored as numpy array
    """
    return face_recognition.load_image_file(image_path)


def face_image_to_encoding(rgb_image):
    """ Find the face in an image and extract its encoding
    Parameters
    ----------
    rgb_image : numpy.darray
        RGB image stored as numpy array
    Returns
    -------
    numpy.darray
        Encoding stored as an array of 128 values
    """
    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=0, model='hog')
    assert len(face_locations) > 0, "Found no face, please use a different image"
    encoding = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=0, model='small')
    if len(encoding) > 1:
        print('Warning: Found more than 1 face, the first one will be used.')
        
    return encoding[0]


def face_image_to_encodings(rgb_image):
    """ Find the face in an image and extract its encoding
    Parameters
    ----------
    rgb_image : numpy.darray
        RGB image stored as numpy array
    Returns
    -------
    numpy.darray
        Encoding stored as an array of 128 values
    """
    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=0, model='hog')
    assert len(face_locations) > 0, "Found no face, please use a different image"
    encoding = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=0, model='small')
    #if len(encoding) > 1:
    #   print('Warning: Found more than 1 face, the first one will be used.')
        
    return encoding


def compute_face_distances(encodings, new_encoding):
    """ Compute Euclidean distance between a new encoding and all encodings.
    Parameters
    ----------
    encodings : list[numpy.darray]
        List of encodings
    new_encoding : numpy.darray
        Encoding stored as an array of 128 values
    Returns
    -------
    list
        List of Eucliden distance between encodings & new encoding
    """
    return face_recognition.face_distance(encodings, new_encoding)


def get_most_similar_face(names, face_distances):
    """ Find the smallest distance & the respective name
    Parameters
    ----------
    names : list
        List of names
    face_distances : list
        List of distances
    Returns
    -------
    (float, str)
        Distance & name of the face that has smallest distance
    """
    #min_idx = np.argmin(face_distances)
    min_idx = np.argmin(names)

    return face_distances[min_idx], names[min_idx]