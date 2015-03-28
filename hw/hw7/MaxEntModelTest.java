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
        
        for(String[] raw_feats : feat_matrix)
        {
            if(feats != null)
            {   
                String tag = model.getBestOutcome(model.eval(feats));
                results.add(tag);
            }
            else
            {
                results.add(null);
            }
        }
        
        return results;
    }
    
    
    private static String[] add_prevBioFeat(String[] feats, String tag)
    {
        String[] ext_feats = new String[feats.length + 1];
        String new_feat = "prevBIO="+tag;
        ext_feats[ext_feats.length - 1] = new_feat;
        return ext_feats;
    }

   private static int find_which_max(double[] arr)
   {
       for(int i = 0, double val = 0.0, int max_ix = 0; i < arr.length; i++)
       {
           if(arr[i] > val)
           {
               val = arr[i];
               max_ix = i;
           }
       }
       
       return max_ix;
       
   }


    //Decode 1 sentence
    public static ArrayList<String> viterbi_decode0(GISModel model, ArrayList<String[]> feat_matrix)
    {
        
        int n_states = model.getNumOutcomes() + 1; //We add 1 more state, for null
        int n_obs = feat_matrix.size(); 
        
        double[] v[n_states][n_obs];
        int[]    p[n_states][n_obs];
                 
        //Initial probability
        double[] estimates = model.eval(add_prevBioFeat(feat_matrix[0], model.getOutcome(n_states)));
       
        for(int i = 0; i < n_states; i++)
        {
            v[i][0] = estimates[i];
        }
        
        //Recursive steps
        for(int j = 1; j < feat_matrix.size(); j++)
        { //for each observation 
            //create a matrix to hold possible values, one row per possible prior state
            ArrayList<double[]> estimateMatrix = new ArrayList<double[]>(n_states); 
            
            for(int i = 0; i < n_states; i++)
            { //for each possible prior state
                double v_prev = v[i][j - 1];
                estimates = model.eval(add_prevBioFeat(feat_matrix[j], model.getOutcome(i)));
                estimateMatrix.add(estimates);
            }
            
            //Find max and store
            double[][] step_result = find_max_matrix(estimateMatrix);
            v[][j].copyarray(step_result[0]);
            p[][j].copyarray(step_result[0]);
        }
        
        //Decode
        ArrayList<String> results = new ArrayList<String>();
        int back_pointer = find_which_max(p[][n_obs - 1]);
        results.add(back_pointer);
        
        for(int j = n_obs - 2; j > 0; j--)
        {
            back_pointer = p[j][back_pointer];
            results.add(back_pointer);
        }
        
        return Collections.reverse(results);
        
    }
    
    
    
    
    public static double simple_accuracy(ArrayList<String> predicted, ArrayList<String> realized) { 
       
        if(predicted.size() != realized.size())
        {
            System.out.println("predicted and realized array lists have different lengths");
            return 0.0;
        }
        else
        {
            int acc = 0, len = 0;
            
            for(int i = 0; i < predicted.size(); i++)
            {
                if(predicted.get(i) != null) //don't consider empty lines in calc
                {
                    len++;
                   
                    if(predicted.get(i).equals(realized.get(i)))
                    {
                        acc++;
                    }
                }

            }
            
            return ((double) acc) / ((double) len);
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
                    realized_tags.add(null); //add a marker for boundary
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


