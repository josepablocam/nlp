import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.io.File;

import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;


class MaxEntModelTest {
    
    public static ArrayList<String> simple_decode(GISModel model, ArrayList<String[]> feat_matrix)
    { 
        ArrayList<String> results = new ArrayList<String>();
        
        for(String[] feats : feat_matrix)
        {
            if(feats != null)
            {
                String tag = model.getBestOutcome(model.eval(feats));
                results.add(tag);
            }
        }
        
        return results;
    }

    //TODO: public static String[] viterbi_decode
    
    
    public static double simple_accuracy(ArrayList<String> predicted, ArrayList<String> realized) { 
       
        if(predicted.size() != realized.size())
        {
            System.out.println("predicted and realized array lists have different lengths");
            return 0.0;
        }
        else
        {
            int acc = 0;
            
            for(int i = 0; i < predicted.size(); i++)
            {
                if(predicted.get(i).equals(realized.get(i)))
                {
                    acc++;
                }
            }
            
            return ((double) acc) / ((double) predicted.size());
        }  
    }
    
    public static void usage()
    {
        System.out.println("MaxEndModelTest usage: gis_model_file feature_txt_file");
    }
    
    public static void main(String[] argv)
    {
        //Test to make sure that command line options are ok
        if(argv.length != 2)
        {
            MaxEntModelTest.usage();
            System.exit(1);
        }
        
        String model_file_name = argv[0];
        String feat_file_name = argv[1];
        
        //read in model
        GISModel model = null;
        
        try {
            model = (GISModel) (new SuffixSensitiveGISModelReader(new File(model_file_name))).getModel();
        } catch(FileNotFoundException e) {
            System.out.println("Unable to open:" + model_file_name);
            System.exit(1);
        } catch(IOException e) {
            System.out.println("Error:" + e);
            System.exit(1);
        }
        
        //read text file of features
        ArrayList<String[]> feature_matrix = new ArrayList<String[]>();
        ArrayList<String> realized_tags = new ArrayList<String>();
        
        try {
            FileReader fileReader = new FileReader(feat_file_name);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line = null;
            
            while((line = bufferedReader.readLine()) != null)
            {
                if(line.trim().length() > 0 )
                {
                    String[] features_raw = line.split(" ");
                    realized_tags.add(features_raw[features_raw.length - 1]); //save tag
                    String[] features = new String[features_raw.length - 1];
                    System.arraycopy(features_raw, 0, features, 0, features.length); 
                    feature_matrix.add(features);
                }
                else
                {
                    feature_matrix.add(null); //add a marker for boundary
                }

            }
            
            bufferedReader.close();
            
        } catch(FileNotFoundException e) {
            System.out.println("Unable to open:" + feat_file_name);
            System.exit(1);
        } catch(IOException e) {
            System.out.println("Error:" + e);
            System.exit(1);
        }
        
        
        //model predictions
        ArrayList<String> predicted_tags = simple_decode(model, feature_matrix); 
        //Model accuracy
        double accuracy = simple_accuracy(predicted_tags, realized_tags);
        
        //write out results to std out
       System.out.println("Accuracy: " + accuracy);
        for(String tag : predicted_tags)
        {
            System.out.println(tag);
        }
        
        
        
    }
    
    
}


