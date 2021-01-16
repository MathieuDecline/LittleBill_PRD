# -*- coding: utf-8 -*-


#Extract label and text from labeled dataset
#Input : PATH to Labeled_Text_Dataset
#Output : TextList , LabelList
def Duplicate_Text2List(PATH): 
    file = open(PATH)
    file = file.read()
    ligne =[]
    label=[]
    texte_label =[]
    texte = []
    file =file.split('\n')
    for i in range (0,len(file)-1,1):
        file[i] = file[i].split(',')
        ligne =  file[i][0]
        label = file[i][1]
        texte_label.append(label)
        texte.append(ligne)
    return texte, texte_label
    file.close()

#Write dataset as text file from a TextList and LabelList
#Input : OutputFileName , TextList, LabelList
#Output : DatasetFile
def WritingFileSentence_non_duplicateLines(file_name, LIST, LABEL_LIST):
    sentence=[]
    List_sentence = (sentence)
    f= open(file_name,"w+")
    #Middle_CONTENT
    for j in range(0, len(LIST), 1):
        sentence=LIST[j]
        List_sentence.append(sentence)
    for m in range(0,len(LIST)-2,1):
        f.write(List_sentence[m]+' ')
        f.write(','+LABEL_LIST[m])
        f.write('\n') 
    return f.close()

#Remove specified label and reshape dataset
#Input : InputFileName, OutputFileName
#Output : DatasetFile (label removed)
def removeUNK(Input_file, Output_file):
    Pos_0=[]
    LIST, LABEL_LIST = Duplicate_Text2List(Input_file)
    print(LABEL_LIST)
    print(len(LABEL_LIST))
    LEN = len(LABEL_LIST)
    print(LEN)
    for i in reversed(range (1, LEN-1, 1)):
        if LABEL_LIST[i] == "0" :
            Pos_0.append(i)
            unk = LABEL_LIST.pop(i)
            text_unk = LIST.pop(i)
            print("LABEL", len(LABEL_LIST))
            print(unk)
            print(i)
    WritingFileSentence_non_duplicateLines(Output_file, LIST, LABEL_LIST)
    return LABEL_LIST

# ==============================================================

print("START")

Input_File = 'sentence_duplicated_doc_text.txt'
Output_File = 'removed_duplicated_0.txt'

removeUNK(Input_File, Output_File)


print("END")    
            
        
