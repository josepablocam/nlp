import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
    
    
class FeatureExtractor {

/*
   // private static String[] pos_and_bio(String[] lines)
    //{
        
    //    ArrayList<String> result = new ArrayList<String>();
     //   String[] components = new String[3];
        
        
        for(String line : lines)
        {
            if(line.trim().length() > 0)
            {
                components = line.split(" ");
                result.add(String.format("%s=%s %s", "pos", components[1], components[2]));                
            }
            else
            {
                result.add("\t");
            }
        }
        
        return result.toArray(new String[result.size()]);
    }
    */
    private static String EMPTY_LINE = "\t";
    
    //Is first letter capitalized or entire word capitalized
    private static String capInfo(String line)
    {
        String word = line.split(" ")[0];
        return "firstCaps="+Character.isUpperCase(word.charAt(0)) + " " + "allCaps="+word.toUpperCase().equals(word);
    }
    
    //contextual POS information
    private static String context_POS(String prev_line, String curr_line, String next_line)
    {
       String prev_pos = null, curr_pos = curr_line.split(" ")[1], next_pos = null; 
       
       if(prev_line != null)
       {
          prev_pos = prev_line.split(" ")[1]; 
       }
       
       if(next_line != null)
       {
           next_pos = next_line.split(" ")[1];
       }
       
       return "prevPOS=" + prev_pos + " " + "currPOS=" + curr_pos + " " + "nextPOS=" + next_pos;  
    }
    
    private static String simple_POS(String curr_line)
    {
        String curr_pos = curr_line.split(" ")[1];
        return "POS=" + curr_pos;
    }
    
    private static String prev_BIO(String prev_line)
    {
        String prev_bio = null;
        
        if(prev_line != null)
        {
            String[] components = prev_line.split(" ");
            
            if(components.length == 3)
            { //otherwise we're dealing with test data with no tags
                prev_bio = components[2]; 
            }
        }
        
        return "prevBIO=" + prev_bio;
    }
    
    
    
    private static ArrayList<String> create_feature_matrix(String[] lines, String feat_type)
    {
        String prev_line = null, next_line = null, curr_line = null; //contextual information
        ArrayList<String> feature_matrix = new ArrayList<String>();
        
        for(int i = 0; i < lines.length; i++, prev_line = curr_line)
        {
            curr_line = lines[i];
            
            if(curr_line.trim().length() > 0)
            {
                if(i + 1 > lines.length - 1 || lines[i + 1].trim().length() == 0)
                {
                    next_line = null;
                }
                else
                {
                    next_line = lines[i + 1];
                }
            
            String feat1 = capInfo(curr_line);
            String feat2 = context_POS(prev_line, curr_line, next_line);
            String obs_tag = curr_line.split(" ")[2];

            if(feat_type.equals("-simple")) {
                feature_matrix.add(feat1 + " " + feat2 + " " + obs_tag);
            } else {
                String feat3 = prev_BIO(prev_line); //for viterbi
                feature_matrix.add(feat1 + " " + feat2 + " " + feat3 + " " + obs_tag);
            }
                  
            }
            else
            { //we hit an empty line
                curr_line = null;
                feature_matrix.add(EMPTY_LINE);
            }
        }
        
        return feature_matrix;
    }
    
    
    

    public static void usage() {
        System.out.println("FeatureExtractor usage: annotated_file [-simple|-viterbi]");
    }
    
    public static void main(String[] argv)
    {
        if(argv.length != 2)
        {
            FeatureExtractor.usage();
            System.exit(1);
        }
        
        String file_name = argv[0];
        ArrayList<String> list = new ArrayList<String>();
        
        try {
            FileReader fileReader = new FileReader(file_name);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line = null;
        
            while ((line = bufferedReader.readLine()) != null) {
            list.add(line);
            }

            bufferedReader.close();
            
         } catch(Exception e) {
             System.out.println("Woopsie");
         }
         
        String[] lines = list.toArray(new String[list.size()]);
        
        ArrayList<String> feature_matrix = create_feature_matrix(lines, argv[1]);
        
        for(String feature_vector : feature_matrix)
        {
            System.out.println(feature_vector);
        }
         
    }
    
}