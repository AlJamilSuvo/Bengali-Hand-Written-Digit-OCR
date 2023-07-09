import os


cwd = os.getcwd()

dirlist = os.listdir(cwd+"/test/")
annotation = ''

for dir in dirlist:

    flist = os.listdir(cwd+"/test/"+dir)
    for filename in flist:
        annotation += 'test/'+dir+'/'+filename+','+dir+'\n'
with open('annotation_test.txt', 'w') as f:
    f.write(annotation)
f.close()
