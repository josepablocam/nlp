import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;



import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;


class MaxEntModelPredict {
    
    //A simple decoding method that commits to a tag at time i, rather than
    //performing a dynamic search like viterbi
    public static ArrayList<String> greedy_decode(GISModel model, ArrayList<String[]> feat_matrix)
    { 
        ArrayList<String> results = new ArrayList<String>();
        String tag = null;
        String prev_tag = "START";
        String prev_tag2 = "START";
        
        
        for(String[] raw_feats : feat_matrix)
        {
            if(raw_feats != null)
            {   
                ArrayList<String> ext_feats = new ArrayList<String>(Arrays.asList(raw_feats));
        
                //add 2 prior tags as features
                ext_feats.add("tag_i-1=" + prev_tag);
                ext_feats.add("tag_i-2=" + prev_tag2);
                
                //predict and store
                String[] feats = ext_feats.toArray(new String[ext_feats.size()]);
                tag = model.getBestOutcome(model.eval(feats));
                results.add(tag);
                
                //move tags to account for i
                prev_tag2 = prev_tag;
                prev_tag = tag;
            }
            else
            {
                results.add(null);
            }
        }
        
        return results;
    }
    
    
   private static double max(double[] probs){
       double max_val = 0.0;

       for(int i = 0; i < probs.length; i++){
           if(probs[i] > max_val){
               max_val = probs[i];
           }
       }

       return max_val;
   }
   
   private static int which_max(double[] probs){
       double max_val = 0.0;
       int max_ix = 0;
       
       for(int i = 0; i < probs.length; i++){
           if(probs[i] > max_val){
               max_val = probs[i];
               max_ix = i;
           }
       }
       
       return max_ix;
   }

    //Decode 1 sentence using viterbi method
    public static ArrayList<String> viterbi_decode0(GISModel model, ArrayList<String[]> feat_matrix){
        int n_states = model.getNumOutcomes();
        int n_obs = feat_matrix.size();
        int n_feats = feat_matrix.get(0).length;

        double[][][] pi = new double[n_obs][n_states + 1][n_states + 1]; //viterbi probabilities, add 1 state for START
        int[][][]    bp = new int[n_obs][n_states + 1][n_states + 1]; //viterbi backpointer, add 1 state for START
        
        String[] labels = new String[n_states + 1];
        labels[labels.length - 1] = "START"; //put start special symbol at end
        
        for(int i = 0; i < labels.length - 1; i++) {
            labels[i] = model.getOutcome(i); //fill in rest of labels
        }
        
        String[] raw_feats = feat_matrix.get(0); //features for first word in sentence
        String[] ext_feats = new String[n_feats + 2]; //add 2 spaces for previous 2 tags
        System.arraycopy(raw_feats, 0, ext_feats, 0, n_feats);
        
        ext_feats[n_feats] = "tag_i-2=START";
        ext_feats[n_feats + 1] = "tag_i-1=START";
        
        //initial probabilities
        System.out.println("Initializing probabilities");
        double[] init_probs = model.eval(ext_feats);
        for(int v = 0; v < n_states; v++) {
            pi[0][n_states][v] = init_probs[v]; //v[0]['START'][v]
            bp[0][n_states][v] = n_states; //bp[0]['START'][v] = 'START'
        }
        
        //recursive step
        //TODO: implement prune        
        System.out.println("Recursive step");
        for(int i = 1; i < n_obs; i++) { //for each observation
            System.out.println("index:" + i);
            System.arraycopy(feat_matrix.get(i), 0, ext_feats, 0, n_feats);
            for(int v = 0; v < n_states; v++){ //tag at i
                for(int u = 0; u < n_states; u++){ //tag at i - 1
                    ext_feats[n_feats + 1] = "tag_i-1=" + labels[u];
                    int w_lim = (i == 1) ? n_states + 1 : n_states; //+1 to allow for 'START' when i = 1
                    double[] probs = new double[w_lim];
                    for(int w = 0; w < w_lim; w++) {
                        ext_feats[n_feats] = "tag_i-2=" + labels[w];
                        if(pi[i-1][w][u] > 0){
                            probs[w] = model.eval(ext_feats)[v] * pi[i - 1][w][u];
                        }
                    }
                    pi[i][u][v] = max(probs);
                    bp[i][u][v] = which_max(probs);
                }
            }
        }
    

        //Decode
        System.out.println("decoding");
        ArrayList<String> results = new ArrayList<String>();
        
        double max_prob = 0.0;
        int pred_w = 0, pred_u = 0, pred_v = 0;
        
        //find last 2 states
        for(int v = 0; v < n_states; v++){
            for(int u = 0; u < n_states; u++){
                if(pi[n_obs - 1][u][v] > max_prob){
                    max_prob = pi[n_obs - 1][u][v];
                    pred_u = u;
                    pred_v = v;
                }
            }
        }
        
        results.add(model.getOutcome(pred_v));
        results.add(model.getOutcome(pred_u));
        
        //trace back based on back pointer
        for(int j = n_obs - 1; j >= 2; j--){ //we already decoded last position
            pred_w = bp[j][pred_u][pred_v];
            results.add(model.getOutcome(pred_w));
            pred_v = pred_u;
            pred_u = pred_w;
        }
        Collections.reverse(results); //reverse backpointer list to get in order
        for(String tag: results) {
            System.out.println(tag);
        }
        
        
        return results;
    }

    
    //decode an entire document using viterbi,calls viterbi_decode0 on each sentence
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
        if(sentence_features.size() != 0) {
            results.addAll(viterbi_decode0(model, sentence_features));
        }
        
        return results;
    }
    

    public static void usage()
    {
        System.out.println("MaxEndModelTest usage: <gis_model_file> <feature_txt_file> <result_file> <-greedy|-viterbi>");
    }
    
    public static void main(String[] argv)
    {
        //Test to make sure that command line options are ok
        if(argv.length != 4)
        {
            MaxEntModelPredict.usage();
            System.exit(1);
        }
        
        String model_file_name = argv[0];
        String feat_file_name = argv[1];
        String result_file_name = argv[2];
        
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
        
        try {
            FileReader fileReader = new FileReader(feat_file_name);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line = null;
            
            while((line = bufferedReader.readLine()) != null)
            {
                if(line.trim().length() > 0 )
                {
                    String[] features = line.split(" ");
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
        ArrayList<String> predicted_tags = new ArrayList<String>();
        if(argv[3].equals("-greedy"))
         {
            predicted_tags = greedy_decode(model, feature_matrix); 
        } else if (argv[3].equals("-viterbi")){
            predicted_tags = viterbi_decode(model, feature_matrix);
        } else {
            System.out.println("method " + argv[3] + " nyi");
            System.exit(1);
        }
            
        //Write out tags to the file name provided
        try {
            PrintWriter writer = new PrintWriter(result_file_name, "UTF-8");

            for(String tag : predicted_tags){
                if(tag != null){
                    writer.println(tag);
                } else {
                    writer.println();
                }
            }

            writer.close();

        } catch(IOException e) {
            System.out.println("Error:" + e);
            System.exit(1);
        }

            
    }
    
    
}


