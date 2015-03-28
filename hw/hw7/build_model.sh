#Make sure all components are compiled freshly
die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 2 ] || die "2 arguments required, $# provided. Need training and test data abs paths"

CLASSPATH=/usr/local/jet/jet-all.jar:.
TRAIN_DATA=$1
TEST_DATA=$2
TRAIN_FEATURES="features_$1"
TEST_FEATURES="features_$2"
MODEL="model_$1.txt"
TRAIN_EVAL="results_$1.txt"
TEST_EVAL="results_$2.txt"

echo 'Compiling tools'
#compile feature extractor
javac -cp $CLASSPATH FeatureExtractor.java || die "Failed to compile FeatureExtractor"
#compile learner
javac -cp $CLASSPATH MaxEntModelTrain.java || die  "Failed to compile MaxEntModelTrain"
#compile tester
javac -cp $CLASSPATH MaxEntModelTest.java  || die "Failed to compile MaxEntModelTest"

#Extract features and save down
#echo 'Extracting features for $1' 
java -cp $CLASSPATH FeatureExtractor $TRAIN_DATA > $TRAIN_FEATURES
#Do same for test data if different source
if [ $1 != $2 ] 
then
    echo "Extracting features for $2"
    java -cp $CLASSPATH FeatureExtractor $TEST_DATA > $TEST_FEATURES
fi

#Train model
echo 'Training model'
java -cp $CLASSPATH MaxEntModelTrain $TRAIN_FEATURES $MODEL

#Test model on training data and save results
echo "Testing model on $1"
java -cp $CLASSPATH MaxEntModelTest $MODEL $TRAIN_FEATURES > $TRAIN_EVAL

if [ $1 != $2 ] 
then
    echo "Testing model on $2"
    java -cp $CLASSPATH MaxEntModelTest $MODEL $TEST_FEATURES > $TEST_EVAL
fi



 
