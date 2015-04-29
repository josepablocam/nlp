import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;
import java.lang.Math;



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
        int n_feats = feat_matrix.get(0).length;
        String[] ext_feats = new String[n_feats + 2]; //we will add 2 more features, the previous 2 tags!
        
        for(String[] raw_feats : feat_matrix)
        {
            if(raw_feats != null)
            {   
                System.arraycopy(raw_feats, 0, ext_feats, 0, n_feats);
        
                //add 2 prior tags as features
                ext_feats[n_feats] = "tag_i-2=" + prev_tag2;
                ext_feats[n_feats + 1] = "tag_i-1=" + prev_tag;
                
                //predict and store
                tag = model.getBestOutcome(model.eval(ext_feats));
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
   

    //Decode 1 sentence using viterbi method
    public static ArrayList<String> viterbi_decode0(GISModel model, ArrayList<String[]> feat_matrix, double beam){
        int n_states = model.getNumOutcomes();
        int n_obs = feat_matrix.size();
        int n_feats = feat_matrix.get(0).length;

        double[][][] pi = new double[n_obs][n_states + 1][n_states + 1]; //viterbi probabilities, add 1 state for START
        int[][][]    bp = new int[n_obs][n_states + 1][n_states + 1]; //viterbi backpointer, add 1 state for START
        
        //initialize pi to negative infinity for all states
        for(int i = 0; i < n_obs; i++) {
            for(int u = 0; u < n_states + 1; u++) {
                Arrays.fill(pi[i][u], Double.NEGATIVE_INFINITY);
            }
        }
        
        //Array of labels with extra labels added
        String[] labels = new String[n_states + 1];
        labels[labels.length - 1] = "START"; //put start special symbol at end
        
        for(int i = 0; i < labels.length - 1; i++) {
            labels[i] = model.getOutcome(i); //fill in rest of labels
        }
        
        //Extended features for first observation
        String[] raw_feats = feat_matrix.get(0); //features for first word in sentence
        String[] ext_feats = new String[n_feats + 2]; //add 2 spaces for previous 2 tags
        System.arraycopy(raw_feats, 0, ext_feats, 0, n_feats);
        
        ext_feats[n_feats] = "tag_i-2=START";
        ext_feats[n_feats + 1] = "tag_i-1=START";
        
        //initial probabilities
        double[] init_probs = model.eval(ext_feats); //probability of each v given START,START
        double beam_base = Double.NEGATIVE_INFINITY; //start out as lowest possible value
            
        for(int v = 0; v < n_states; v++) {
            double curr_prob = Math.log10(init_probs[v]); //v[0]['START'][v]
            pi[0][n_states][v] = curr_prob;
            bp[0][n_states][v] = n_states; //bp[0]['START'][v] = 'START'
            
            if(curr_prob > beam_base) {
                beam_base = curr_prob;
            }
        }
        
        double log_beam = Math.log10(beam);
        
        for(int i = 1; i < n_obs; i++) { //for each observation
            System.arraycopy(feat_matrix.get(i), 0, ext_feats, 0, n_feats); //copy the features to our extended feature vector
            int w_lim = (i == 1) ? n_states + 1 : n_states; //+1 to allow for 'START' when i = 1
            double[][][] probs = new double[w_lim][n_states][n_states]; //probs[w][u][v]
            double curr_max = Double.NEGATIVE_INFINITY; //maximum probability associated with a sequence up to obs i
            
            for(int u = 0; u < n_states; u++){ //tag at i - 1
                ext_feats[n_feats + 1] = "tag_i-1=" + labels[u];
                for(int w = 0; w < w_lim; w++) {
                    ext_feats[n_feats] = "tag_i-2=" + labels[w];
                    if(pi[i - 1][w][u] >= beam_base + log_beam){
                        //only worth calculating model based on history:w,u if pi value is within beam
                        System.arraycopy(model.eval(ext_feats), 0, probs[w][u], 0, n_states); //vector of P(v|w,u) for each v
                        for(int v = 0; v < n_states; v++) { //iterate through predicted values
                            double curr_prob = Math.log10(probs[w][u][v]) + pi[i - 1][w][u]; //note must convert model probs to log base
                            if(curr_prob > pi[i][u][v]) { //find max and argmax for u,v
                                pi[i][u][v] = curr_prob;
                                bp[i][u][v] = w;
                            }
                            //used to reset beam_base
                            if(curr_prob > curr_max){
                                curr_max = curr_prob;
                            }
                        }
                    }
                } 
            }
            
            beam_base = curr_max;
        }
    

        //Decode
        ArrayList<String> results = new ArrayList<String>();
        
        double max_prob = Double.NEGATIVE_INFINITY;
        int pred_w = 0, pred_u = 0, pred_v = 0;
        
        //find last 2 states
        for(int v = 0; v < n_states; v++){
            for(int u = 0; u < n_states; u++){
                double curr_prob = pi[n_obs - 1][u][v];
                if(curr_prob > max_prob){
                    max_prob = curr_prob;
                    pred_u = u;
                    pred_v = v;
                }
            }
        }
        
        //resolve the actual tag names before adding, by using the labels array
        results.add(labels[pred_v]);
        results.add(labels[pred_u]);
        
        //trace back based on back pointer
        for(int j = n_obs - 1; j >= 2; j--){ //we already decoded last position
            pred_w = bp[j][pred_u][pred_v];
            results.add(labels[pred_w]);
            pred_v = pred_u;
            pred_u = pred_w;
        }
        Collections.reverse(results); //reverse backpointer list to get in order
        return results;
    }

    
    //decode an entire document using viterbi, calls viterbi_decode0 on each sentence
    public static ArrayList<String> viterbi_decode(GISModel model, ArrayList<String[]> feat_matrix, double beam) {
        ArrayList<String> results = new ArrayList<String>();
        String[] word_features = null;
        ArrayList<String[]> sentence_features = new ArrayList<String[]>();
        int sent_ct = 0;
        int feat_matriz_size = feat_matrix.size();
        
        
        for(int i = 0; i < feat_matriz_size; i++){
            word_features = feat_matrix.get(i);

            if(word_features == null) {
                //Decode 1 sentence, we reached a sentence boundary  
                ArrayList<String> sent_results = viterbi_decode0(model, sentence_features, beam);
                results.addAll(sent_results);
                results.add(null); //add an empty result to mark boundary
                sentence_features = new ArrayList<String[]>();  //create new container for next sentence to use
                sent_ct++;
                System.out.println("Processed: " + sent_ct + " sentences"); //let user know how many sentences so far, in case too slow
            } else {
                sentence_features.add(word_features);
            }
        }
        //The loop above could miss a decode if there is no null at the end
        if(sentence_features.size() != 0) {
            results.addAll(viterbi_decode0(model, sentence_features, beam));
        }
        
        return results;
    }
    

    public static void usage()
    {
        System.out.println("MaxEndModelTest usage: <gis_model_file> <feature_txt_file> <result_file> <-greedy|-viterbi [beam_factor]>");
    }
    
    public static void main(String[] argv)
    {
        //Test to make sure that command line options are ok
        if(argv.length < 4)
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
        String decode_method = argv[3];
        
        ArrayList<String> predicted_tags = new ArrayList<String>();
        if(decode_method.equals("-greedy"))
         {
            predicted_tags = greedy_decode(model, feature_matrix); 
            
        } else if(decode_method.equals("-viterbi")){
            //If we don't get a beam size, we assume this means it is
            //simple viterbi...takes a long time though!!!
            double beam_factor = 1.0;
            
            if(argv.length == 5)
            {
                beam_factor = Double.parseDouble(argv[4]); //user provided beam factor, use that
            }
        
            predicted_tags = viterbi_decode(model, feature_matrix, beam_factor);
            
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
                    writer.println(); //blank line separate sentences
                }
            }

            writer.close();

        } catch(IOException e) {
            System.out.println("Error:" + e);
            System.exit(1);
        }

            
    }
    
    
}


