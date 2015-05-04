MODELPATH="./models/"
CP="/usr/local/jet/jet-all.jar:."

all: javatrain javapredict

javatrain: MaxEntModelTrain.java
	javac -cp ${CP} "${MODELPATH}MaxEntModelTrain.java"

javapredict: MaxEntModelPredict.java
	javac -cp ${CP} "${MODELPATH}MaxEntModelPredict.java"

clean:rm "${MODELPATH}.class"

