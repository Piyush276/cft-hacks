from collections import defaultdict as dic
import speech_recognition as sr

h=dic(int)

from monkeylearn import MonkeyLearn
print(" \n \nSPEAK ANYTHINK AND SAY EXIT TO GET OUT OF LOOP\n \n")

input_value="go"

while (input_value!="exit" ):
   ml = MonkeyLearn('4bacc60b35272eb0a59ebc4c4809addf754dfa24')
   data = []
   # input_value=input()
   r = sr.Recognizer()
   with sr.Microphone() as source:
       audio = r.listen(source)
       # audio=str(audio)
   try :
       print(r.recognize_google(audio),"\n")
   except Exception as e:
       print("speak again please \n")
       continue
   # print(r.recognize_google(audio))
   input_value=r.recognize_google(audio)
   data.append(input_value)
   model_id = 'cl_pi3C7JiL'
   result = ml.classifiers.classify(model_id, data)
   part1=result.body[0]['classifications'][0]['tag_name']
   part2=result.body[0]['classifications'][0]['confidence']*100
   print(part1,part2)
   h[part1]+=part2
   print("-------------------")


