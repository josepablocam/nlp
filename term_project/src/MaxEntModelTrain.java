import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.io.File;

import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;


class MaxEntModelTrain {
    public static void usage() {
         System.out.println("MaxEntModelTrain usage: <feature_file> <output_model_file>");
    }
    public static void main (String[] argv) {
        int CUT_OFF = 4;
        int ITERATIONS = 100;
        
        if(argv.length != 2)
        {
            MaxEntModelTrain.usage();
            System.exit(1);
        }

        String feature_file_name = argv[0];
        String model_save_name = argv[1];

        try {
            FileReader datafr = new FileReader(feature_file_name);
            EventStream events = new BasicEventStream(new PlainTextByLineDataStream(datafr));
            GISModel model = GIS.trainModel(events, ITERATIONS, CUT_OFF);
            File outputFile = new File(model_save_name);
            GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
            writer.persist();
            datafr.close();
        } catch (Exception e) {
            System.out.print("Unable to create model due to exception: " + e);
        }
     }
}


