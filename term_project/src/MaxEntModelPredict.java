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
    
    
   // private static int find_which_max(double[][] matrix, int col)
//    {
//        double val = 0.0;
//        int max_ix = 0;
//
//        for(int i = 0; i < matrix.length; i++)
//        {
//            if(matrix[i][col] > val)
//            {
//                val = matrix[i][col];
//                max_ix = i;
//            }
//        }
//
//        return max_ix;
//
//    }
//
//    private static void find_max_viterbi(ArrayList<double[]> matrix, double[][] v, int[][] p, int col)
//    {
//        int n_states = matrix.size();
//        double[] poss;
//        double[] max_probs = new double[n_states];
//        int[] max_ix = new int[n_states];
//
//
//        for(int i = 0; i < n_states; i++)
//        {
//            poss = matrix.get(i);
//
//            for(int j = 0; j < n_states; j++)
//            {
//                if(poss[j] > max_probs[j])
//                {
//                     max_probs[j] = poss[j];
//                     max_ix[j] = i;
//                }
//            }
//        }
//
//        //copy values into our v and p matrices
//        for(int i = 0 ; i < n_states; i++) {
//            v[i][col] = max_probs[i];
//            p[i][col] = max_ix[i];
//        }
//
//    }
//
//
//     //Decode 1 sentence
//     public static ArrayList<String> viterbi_decode0(GISModel model, ArrayList<String[]> feat_matrix)
//     {
//         int n_states = model.getNumOutcomes();
//         int n_obs = feat_matrix.size();
//
//         double[][][] v = new double[n_obs][n_states][n_states]; //viterbi probabilities
//         int[][][]    bp = new int[n_obs][n_states][n_states]; //viterbi backpointer
//
//         //Initial probability
//
//
//         double[] estimates = model.eval(feat_matrix.get(0),"null"));
//         String prev_tag = "START";
//         String prev_tag2 = "START";
//         String w_labels = model.getAllOutcomes();
//         String u_labels =
//
//
//         //Recursive steps
//         for(int i = 0; i < n_obs; i++) { //for each observation
//             ArrayList<String> features = new ArrayList<String[]>();
//             features.addAll(feat_matrix.get(i));
//             for(int v = 0; v < n_states; v++){ //tag at i
//                 for(int u = 0; u < n_states; u++){ //tag at i - 1
//                     features.add("tag_i-1=" + model.getOutcome(u));
//                     double[] probs = new probs[n_states];
//                     for(int w = 0; w < n_states; w++) {
//                         //features for obs i
//                         features.add("tag_i-2=" + model.getOutcome(w));
//                         probs[w] = log(model.eval(features.toArray())) + v[j-1][w][u];
//                         features.remove(features.size() - 1); //remove tag_i-2
//                     }
//                     v[j][u][v] = max(probs);
//                     bp[j][u][v] = /* index of max */
//                     features.remove(features.size() - 1); //remove tag_i-1
//                 }
//         }
//
//         for(int v = 0; v < n_states; v++){
//             for(int u = 0; u < n_states; u++){
//             //termination step
//             }
//         }
//
//         //Decode
//         ArrayList<String> results = new ArrayList<String>();
//         int back_pointer = find_which_max(v, n_obs - 1); //find last state that maximizes probability
//         results.add(model.getOutcome(back_pointer));
//
//         //trace back based on back pointer
//         for(int j = n_obs - 1; j > 0; j--){ //we already decoded last position
//             back_pointer = p[back_pointer][j];
//             results.add(model.getOutcome(back_pointer));
//         }
//         Collections.reverse(results); //reverse backpointer list to get in order
//         return results;
//
//     }
//
//     //decode an entire document using viterbi,calls viterbi_decode0 on each sentence separately
//     public static ArrayList<String> viterbi_decode(GISModel model, ArrayList<String[]> feat_matrix) {
//         ArrayList<String> results = new ArrayList<String>();
//         String[] word_features = null;
//         ArrayList<String[]> sentence_features = new ArrayList<String[]>();
//
//         for(int i = 0; i < feat_matrix.size(); i++){
//             word_features = feat_matrix.get(i);
//
//             if(word_features == null) {
//                 //Decode 1 sentence, we reached a sentence boundary
//                 results.addAll(viterbi_decode0(model, sentence_features));
//                 results.add(null); //add an empty result to mark boundary
//                 sentence_features = new ArrayList<String[]>();  //create new for next sentence to use
//             } else {
//                 sentence_features.add(word_features);
//             }
//         }
//         //The loop above could miss a decode if there is no null at the end
//         if(sentence_features.size() != 0)
//         {
//             results.addAll(viterbi_decode0(model, sentence_features));
//         }
//
//         return results;
//
//     }
    

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
            System.out.println("viterbi nyi");
            System.exit(1);
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


