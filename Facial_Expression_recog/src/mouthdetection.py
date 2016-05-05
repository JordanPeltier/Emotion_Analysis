def findmouth(img, model_face, model_mouth):

  # INITIALIZE: loading the classifiers

  # running the classifiers
  detectedFace = model_face.detectMultiScale(img, minSize = (100, 100))
  detectedMouth = model_mouth.detectMultiScale(img, minSize = (30, 30))

  # FACE: find the largest detected face as detected face
  maxFaceSize = 0
  maxFace = ()
  if detectedFace.any():
   for face in detectedFace: # face: [0]: x; [1]: y; [2]: width; [3]: height
    if face[3]*face[2] > maxFaceSize:
      maxFaceSize = face[3]*face[2]
      maxFace = face

  def mouth_in_lower_face(mouth,face):
    # if the mouth is in the lower 2/5 of the face 
    # and the lower edge of mouth is above that of the face
    # and the horizontal center of the mouth is the center of the face
    if (mouth[1] > face[1] + face[3] * 3 / float(5)
      and mouth[1] + mouth[3] < face[1] + face[3]
      and abs((mouth[0] + mouth[2] / float(2))
        - (face[0] + face[2] / float(2))) < face[2] / float(10)):
      return True
    else:
      return False

  # FILTER MOUTH
  filteredMouth = []
  if detectedMouth.any():
   for mouth in detectedMouth:
    if mouth_in_lower_face(mouth,maxFace):
      filteredMouth.append(mouth) 

  maxMouthSize = 0
  maxMouth = ()
  for mouth in filteredMouth:
    if mouth[3]* mouth[2] > maxMouthSize:
      maxMouthSize = mouth[3]* mouth[2]
      maxMouth = mouth
      

  return maxFace, maxMouth


