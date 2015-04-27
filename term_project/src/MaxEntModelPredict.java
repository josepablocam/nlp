import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;
import java.io.File;


import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;


class MaxEntModelPredict {
    
    public static ArrayList<String> greedy_decode(GISModel model, ArrayList<String[]> feat_matrix)
    { 
        ArrayList<String> results = new ArrayList<String>();
        
        for(String[] feats : feat_matrix)
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
    
    
    private static String[] mod_prevBioFeat(String[] feats, String tag)
    {
        //modify a prevBio=null to the new tag
        String[] mod_feats = new String[feats.length];
    
        
        for(int i = 0; i < feats.length; i++)
        {
            String[] feat_comp = feats[i].split("=");
            
            if(feat_comp[0].equals("prevBIO"))
            {
                mod_feats[i] = "prevBIO=" + tag;
            }
            else
            {
                mod_feats[i] = feats[i];
            }
        }
        
        return mod_feats;
        
    }

   private static int find_which_max(double[][] matrix, int col)
   {
       double val = 0.0;
       int max_ix = 0;
       
       for(int i = 0; i < matrix.length; i++)
       {
           if(matrix[i][col] > val)
           {
               val = matrix[i][col];
               max_ix = i;
           }
       }
       
       return max_ix;
       
   }
   
   private static void find_max_viterbi(ArrayList<double[]> matrix, double[][] v, int[][] p, int col)
   {
       int n_states = matrix.size();
       double[] poss;
       double[] max_probs = new double[n_states];
       int[] max_ix = new int[n_states];
       
       
       for(int i = 0; i < n_states; i++)
       {
           poss = matrix.get(i);
           
           for(int j = 0; j < n_states; j++)
           {
               if(poss[j] > max_probs[j])
               {
                    max_probs[j] = poss[j];
                    max_ix[j] = i;
               }
           }
       }
       
       //copy values into our v and p matrices
       for(int i = 0 ; i < n_states; i++) {
           v[i][col] = max_probs[i];
           p[i][col] = max_ix[i];
       } 
       
   }


    //Decode 1 sentence
    public static ArrayList<String> viterbi_decode0(GISModel model, ArrayList<String[]> feat_matrix)
    {
        
        int n_states = model.getNumOutcomes();
        int n_obs = feat_matrix.size(); 
        
        String[] mod_feats;
        double[][] v = new double[n_states][n_obs]; //viterbi probabilities
        int[][]    p = new int[n_states][n_obs]; //viterbi backpointer
                 
        //Initial probability
        double[] estimates = model.eval(mod_prevBioFeat(feat_matrix.get(0),"null"));

        for(int i = 0; i < n_states; i++)
        {
            v[i][0] = estimates[i];
        }

        //Recursive steps
        for(int j = 1; j < n_obs; j++)
        { //for each observation 
            //create a matrix to hold possible values, one row per possible prior state
            ArrayList<double[]> estimateMatrix = new ArrayList<double[]>(); 
            
            for(int i = 0; i < n_states; i++)
            { //for each possible prior state: P(s|s', o)
                double v_prev = v[i][j - 1];
                mod_feats = mod_prevBioFeat(feat_matrix.get(j), model.getOutcome(i));
                estimates = model.eval(mod_feats);
                
                //multiply by previous viterbi prob: P(s|s',o) * v_t-1(s')
                for(int k = 0; k < n_states; k++) {
                    estimates[k] = estimates[k] * v_prev;
                }
                //add to estimate matrix
                estimateMatrix.add(estimates);
            }
            
            //Find max for each possible state and store
            find_max_viterbi(estimateMatrix, v, p, j);
            
        }
        
        //Decode
        ArrayList<String> results = new ArrayList<String>();
        int back_pointer = find_which_max(v, n_obs - 1); //find last state that maximizes probability
        results.add(model.getOutcome(back_pointer));
        
        //trace back based on back pointer
        for(int j = n_obs - 1; j > 0; j--){ //we already decoded last position
            back_pointer = p[back_pointer][j];
            results.add(model.getOutcome(back_pointer));
        }
        Collections.reverse(results); //reverse backpointer list to get in order
        return results;
        
    }
    
    //decode an entire document using viterbi,calls viterbi_decode0 on each sentence separately
    public static ArrayList<String> viterbi_decode(GISModel model, ArrayList<String[]> feat_matrix) {
        ArrayList<String> results = new ArrayList<String>();
        String[] word_features = null;
        ArrayList<String[]> sentence_features = new ArrayList<String[]>();
        
        for(int i = 0; i < feat_matrix.size(); i++){
            word_features = feat_matrix.get(i);
            
            if(word_features == null) {
                //Decode 1 sentence, we reached a sentence boundary
                results.addAll(viterbi_decode0(model, sentence_features)); 
                results.add(null); //add an empty result to mark boundary
                sentence_features = new ArrayList<String[]>();  //create new for next sentence to use
            } else {
                sentence_features.add(word_features);
            }
        }
        //The loop above could miss a decode if there is no null at the end
        if(sentence_features.size() != 0)
        {
            results.addAll(viterbi_decode0(model, sentence_features));
        }
        
        return results;
        
    }
    

    public static void usage()
    {
        System.out.println("MaxEndModelTest usage: <gis_model_file> <feature_txt_file> <-simple|-viterbi>");
    }
    
    public static void main(String[] argv)
    {
        //Test to make sure that command line options are ok
        if(argv.length != 3)
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
                    System.arraycopy(features_raw, 0, features, 0, features.length); //everything except the tag
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
        ArrayList<String> predicted_tags;
        if(argv[2].equals("-simple"))
         {
            predicted_tags = simple_decode(model, feature_matrix); 
        } else {
            predicted_tags = viterbi_decode(model, feature_matrix);
        }
    
        //Model Evaluation
        ArrayList<String> realized_boundaries = get_boundaries(realized_tags);
        ArrayList<String> predicted_boundaries = get_boundaries(predicted_tags);
        
        double tag_accuracy = simple_accuracy(predicted_tags, realized_tags);
        double boundary_precision = pct_in(predicted_boundaries, realized_boundaries);
        double boundary_recall = pct_in(realized_boundaries, predicted_boundaries);
        double boundary_f1 = 2 * ((boundary_precision * boundary_recall) / (boundary_precision + boundary_recall));
        
        //write out evaluation results to std out
        System.out.println("Tag Accuracy: " + tag_accuracy);
        System.out.println("Boundary Precision: " + boundary_precision);
        System.out.println("Boundary Recall: " + boundary_recall);
        System.out.println("Boundary F1: " + boundary_f1);
        
        //Write out tags to std out
        for(String tag : predicted_tags)
        {
            if(tag != null) {
                System.out.println(tag);
            } else {
                System.out.println(); //empty line
            }
        }    
    }
    
    
}


