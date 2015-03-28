import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
    
    
class FeatureExtractor {

    private static String[] pos_and_bio(String[] lines)
    {
        
        ArrayList<String> result = new ArrayList<String>();
        String[] components = new String[3];
        
        
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
    

    public static void usage() {
        System.out.println("FeatureExtractor usage: annotated_file");
    }
    
    public static void main(String[] argv)
    {
        if(argv.length != 1)
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
        
        String[] feats = FeatureExtractor.pos_and_bio(lines);
        
        for(String feat:feats)
        {
            System.out.println(feat);
        }
         
    }
    
}