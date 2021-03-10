import pandas as pd
name = "titan"
data = pd.read_csv('titan.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
convert = open("convert","w")
jud = open("1", "w")
docs = open("docid", "w")
with open(name+".svm.fil", "w") as svm:
    for i in range(len(x)):
        #se quiser colocar o código do próprio documento é só modificar essas tres linhas abaixo e trocar o str(i) por str(x[i][0])
        svm.write(str(i))
        docs.write(str(i) + "\n")
        if y[i][0] > 1:
            y[i][0] = 1
        jud.write("1" + "\t" + "0" + "\t" + str(i) + "\t " + str(y[i][0]) + "\n")
        convert.write(str(i) + "\t" + str(x[i][0]) + "\t" + str(y[i][0]) + "\n")
        for j in range(len(x[0])):
            if j==0:
                continue
            if x[i][j] == 0:
                continue
            elif x[i][j] > 1:
                x[i][j] = 1/x[i][j]
            svm.write(" " + str(j) + ":" + str(x[i][j]))
        svm.write("\n")
convert.close()
jud.close()
docs.close()


