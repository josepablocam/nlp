CP="/usr/local/jet/jet-all.jar:."

all:javatrain javapredict

javatrain:./models/MaxEntModelTrain.java
	javac -cp ${CP} ./models/MaxEntModelTrain.java

javapredict:./models/MaxEntModelPredict.java
	javac -cp ${CP} ./models/MaxEntModelPredict.java

clean:rm ./models/*.class

