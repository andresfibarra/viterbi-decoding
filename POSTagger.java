import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class POSTagger {
    private Map<String, Map<String, Double>> posMap;     //Map POS -> (Map POS -> log prob of occurrences);
    private Map<String, Map<String, Double>> posToWordMap;    //Map POS -> (Map observedWord -> log prob of occurrences)
    private final double UNSEEN_SCORE = -20;

    public POSTagger (String tagsFileName, String sentencesFileName){
        posMap = new HashMap<>();
        posToWordMap = new HashMap<>();

        train(tagsFileName, sentencesFileName);
    }

    /**
     * Uses Viterbi decoding to find more probable list of tags for the given sentence
     * @return - List of most likely tags corresponding to the words in the provided text
     */
    public List<String> viterbi (String lineString) {
        Map<Integer, Map<String, String>> backTrack = new HashMap<>(); //map (layer -> map(currTag -> prevTag))
        String[] line = lineString.toLowerCase().split(" ");

        //work forward
        //begin with start state, #
        Set<String> currTags = new HashSet<>();
        currTags.add("#");
        //begin with a score of 0 at start
        Map<String, Double> currScores = new HashMap<>();
        currScores.put("#", 0.0);

        //loop through every word, stopping at the second to last one
        for (int i = 0; i < line.length; i++) {
            Set<String> nextTags = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();
            backTrack.put(i, new HashMap<String, String>());

            //loop through every current tag
            for(String currTag: currTags) {
                //add each possible next State (tag)
                if (posMap.containsKey(currTag)) {
                    for(String nextTag: posMap.get(currTag).keySet()){
                        nextTags.add(nextTag);
                        double observationScore;
                        if (posToWordMap.containsKey(nextTag) && posToWordMap.get(nextTag).containsKey(line[i])) {
                            observationScore = posToWordMap.get(nextTag).get(line[i]);
                        }
                        else observationScore = UNSEEN_SCORE;
                        double nextScore = currScores.get(currTag) + posMap.get(currTag).get(nextTag) + observationScore;
                        if (!nextScores.containsKey(nextTag) || nextScore > nextScores.get(nextTag)) {
                            nextScores.put(nextTag, nextScore);
                            backTrack.get(i).put(nextTag, currTag);
                        }
                    }
                }
            }
            currTags = nextTags;
            currScores = nextScores;
        }

        //backtrack to find best
        int layer = line.length-1;
        String tag = "";
        Stack<String> tags = new Stack<>();

        //get best tag to start viterbi backtrack
        double bestScore = Double.NEGATIVE_INFINITY;
        for (String bestTag: currScores.keySet()) {
            if (currScores.get(bestTag) > bestScore) {
                bestScore = currScores.get(bestTag);
                tag = bestTag;
            }
        }

        while (layer >= 0) {
            tags.push(tag);
            tag = backTrack.get(layer).get(tag); //set tag = its previous pointer
            layer --;
        }
        //return list of tags;
        List<String> s = new ArrayList<String>();
        String tagAddition;
        while(!tags.empty()) {
            s.add(tags.pop());
        }
        return s;
    }

    /**
     * Build occurrence maps
     * @param tagsFileName
     * @param sentencesFileName
     */
    public void train (String tagsFileName, String sentencesFileName) {
        BufferedReader tagsInput;
        BufferedReader sentencesInput;

        //create readers
        try{
            tagsInput = new BufferedReader(new FileReader(tagsFileName));
            sentencesInput = new BufferedReader(new FileReader(sentencesFileName));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file. \n" + e.getMessage());
            return;
        }
        //loop through files and build maps
        try{
            String tagLine;
            String sentLine;

            while ((tagLine = tagsInput.readLine()) != null && (sentLine = sentencesInput.readLine()) != null) {
                String[] tags = tagLine.split(" ");
                String[] words = sentLine.toLowerCase().split(" ");
                //loop through files and add to map
                for (int i = 0; i < tags.length; i++) {
                    //add values to tag map
                    if (i == 0) {
                        if (!posMap.containsKey("#")) {
                            posMap.put("#", new HashMap<String, Double>());
                        }
                        if (!posMap.get("#").containsKey(tags[0])) { //if this is the first time seeing this sequence of tags
                            posMap.get("#").put(tags[0], 1.0);
                        }
                        else {
                            posMap.get("#").put(tags[0], posMap.get("#").get(tags[0])+1);
                        }
                    }
                    else if (!posMap.containsKey(tags[i-1])) {  //if first time seeing the preceding tag
                        posMap.put(tags[i-1], new HashMap<String, Double>());
                        posMap.get(tags[i-1]).put(tags[i], 1.0);
                    }
                    else {
                        if (!posMap.get(tags[i-1]).containsKey(tags[i])) { //if this is the first time seeing this sequence of tags
                            posMap.get(tags[i-1]).put(tags[i], 1.0);
                        }
                        else {
                            posMap.get(tags[i-1]).put(tags[i], posMap.get(tags[i-1]).get(tags[i])+1);
                        }
                    }
                    //add values to POS->word map
                    if (!posToWordMap.containsKey(tags[i])) {
                        posToWordMap.put(tags[i], new HashMap<String, Double>());
                        posToWordMap.get(tags[i]).put(words[i], 1.0);
                    }
                    else {
                        if (!posToWordMap.get(tags[i]).containsKey(words[i])) {     //if this is the first time seeing this sequence of tag --> word
                            posToWordMap.get(tags[i]).put(words[i], 1.0);
                        }
                        else {
                            posToWordMap.get(tags[i]).put(words[i], posToWordMap.get(tags[i]).get(words[i])+1);
                        }
                    }
                }
            }

            //sum totals in posMap and calc log probabilities to update the maps
            for(String pos: posMap.keySet()) { //loop through every tag in posMap
                double count = 0;
                //find the total count of for this tag
                for(String subPos: posMap.get(pos).keySet()) {
                    count += posMap.get(pos).get(subPos);
                }
                //reset each number in subMap to be log((originalNumber)/count)
                for(String subPos: posMap.get(pos).keySet()) {
                    posMap.get(pos).put(subPos, Math.log(posMap.get(pos).get(subPos)/count));
                }
            }
            //repeat for posToWordMap
            for(String pos: posToWordMap.keySet()) { //loop through every tag in posMap
                double count = 0;
                //find the total count of for this word
                for(String word: posToWordMap.get(pos).keySet()) {
                    count += posToWordMap.get(pos).get(word);
                }
                //convert each number in subMap to be log((originalNumber)/count)
                for(String word: posToWordMap.get(pos).keySet()) {
                    posToWordMap.get(pos).put(word, Math.log(posToWordMap.get(pos).get(word)/count));
                }
            }
        }
        catch (IOException e) {
            System.err.println("IO error while reading.\n" + e.getMessage());
        }
        //close files
        finally {
            try {
                tagsInput.close();
                sentencesInput.close();
            }
            catch (IOException e) {
                System.err.println("Cannot close file.\n" + e.getMessage());
            }
        }
    }

    /**
     * tests the model using the specified files
     * @return - list where item at idx0 is the number of correct tags, idx1 is number wrong
     */
    public List<Integer> testModel(String tagsFileName, String sentencesFileName) {
        BufferedReader tagsInput;
        BufferedReader sentencesInput;
        int correct = 0;
        int wrong = 0;

        //create readers
        try{
            tagsInput = new BufferedReader(new FileReader(tagsFileName));
            sentencesInput = new BufferedReader(new FileReader(sentencesFileName));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file. \n" + e.getMessage());
            return new ArrayList<Integer>();
        }
        //loop through files and keep track of successes and failures
        try{
            String tagLine;
            String sentLine;
            List<String> viterbiTagLine;

            while ((tagLine = tagsInput.readLine()) != null && (sentLine = sentencesInput.readLine()) != null) {
                String[] tags = tagLine.split(" ");
                viterbiTagLine = viterbi(sentLine);

                for (int i = 0; i < viterbiTagLine.size(); i++) {
                    if (viterbiTagLine.get(i).equals(tags[i])) {
                        correct++;
                    }
                    else {
                        wrong++;
                    }
                }
            }

        }
        catch (IOException e) {
            System.err.println("IO error while reading.\n" + e.getMessage());
        }
        //close files
        finally {
            try {
                tagsInput.close();
                sentencesInput.close();
            }
            catch (IOException e) {
                System.err.println("Cannot close file.\n" + e.getMessage());
            }
        }
        List<Integer> results = new ArrayList<>();
        results.add(0, correct);
        results.add(1, wrong);
        return results;
    }

    /**
     * Test case from recitation, hardcodes in the training data
     */
    public void test0() {
        //hard code in the occurrence maps
        posMap = new HashMap<>();
        posMap.put("#", new HashMap<String, Double>());
        posMap.get("#").put("NP", 3.0);
        posMap.get("#").put("N", 7.0);
        posMap.put("NP", new HashMap<String, Double>());
        posMap.get("NP").put("CNJ", 2.0);
        posMap.get("NP").put("V", 8.0);
        posMap.put("N", new HashMap<String, Double>());
        posMap.get("N").put("V", 8.0);
        posMap.get("N").put("CNJ", 2.0);
        posMap.put("CNJ", new HashMap<String, Double>());
        posMap.get("CNJ").put("NP", 2.0);
        posMap.get("CNJ").put("V", 4.0);
        posMap.get("CNJ").put("N", 4.0);
        posMap.put("V", new HashMap<String, Double>());
        posMap.get("V").put("NP", 4.0);
        posMap.get("V").put("CNJ", 2.0);
        posMap.get("V").put("N", 4.0);

        posToWordMap = new HashMap<>();
        posToWordMap.put("NP", new HashMap<String, Double>());
        posToWordMap.get("NP").put("chase", 10.0);
        posToWordMap.put("CNJ", new HashMap<String, Double>());
        posToWordMap.get("CNJ").put("and", 10.0);
        posToWordMap.put("V", new HashMap<String, Double>());
        posToWordMap.get("V").put("get", 1.0);
        posToWordMap.get("V").put("chase", 3.0);
        posToWordMap.get("V").put("watch", 6.0);
        posToWordMap.put("N", new HashMap<String, Double>());
        posToWordMap.get("N").put("cat", 4.0);
        posToWordMap.get("N").put("dog", 4.0);
        posToWordMap.get("N").put("watch", 2.0);
        //run viterbi on sample sentence and print it out
        System.out.println(viterbi("chase watch dog chase watch"));
    }

    /**
     * Test case from the PSet instructions, hard codes in the training data
     */
    public void test1() {
        //hard code in the occurrence maps
        posMap = new HashMap<>();
        posMap.put("#", new HashMap<String, Double>());
        posMap.get("#").put("NP", -1.6);
        posMap.get("#").put("MOD", -2.3);
        posMap.get("#").put("PRO", -1.2);
        posMap.get("#").put("DET", -0.9);
        posMap.put("NP", new HashMap<String, Double>());
        posMap.get("NP").put("VD", -0.7);
        posMap.get("NP").put("V", -0.7);
        posMap.put("DET", new HashMap<String, Double>());
        posMap.get("DET").put("N", 0.0);
        posMap.put("VD", new HashMap<String, Double>());
        posMap.get("VD").put("DET", -1.1);
        posMap.get("VD").put("PRO", -0.4);
        posMap.put("N", new HashMap<String, Double>());
        posMap.get("N").put("VD", -1.4);
        posMap.get("N").put("V", -0.3);
        posMap.put("PRO", new HashMap<String, Double>());
        posMap.get("PRO").put("VD", -1.6);
        posMap.get("PRO").put("V", -0.5);
        posMap.get("PRO").put("MOD", -1.6);
        posMap.put("V", new HashMap<String, Double>());
        posMap.get("V").put("DET", -0.2);
        posMap.get("V").put("PRO", -1.9);

        posToWordMap = new HashMap<>();
        posToWordMap.put("NP", new HashMap<String, Double>());
        posToWordMap.get("NP").put("jobs", -0.7);
        posToWordMap.get("NP").put("will", -0.7);
        posToWordMap.put("DET", new HashMap<String, Double>());
        posToWordMap.get("DET").put("a", -1.3);
        posToWordMap.get("DET").put("many", -1.7);
        posToWordMap.get("DET").put("one", -1.7);
        posToWordMap.get("DET").put("the", -1.0);
        posToWordMap.put("VD", new HashMap<String, Double>());
        posToWordMap.get("VD").put("saw", -1.1);
        posToWordMap.get("VD").put("were", -1.1);
        posToWordMap.get("VD").put("wore", -1.1);
        posToWordMap.put("N", new HashMap<String, Double>());
        posToWordMap.get("N").put("color", -2.4);
        posToWordMap.get("N").put("cook", -2.4);
        posToWordMap.get("N").put("fish", -1.0);
        posToWordMap.get("N").put("jobs", -2.4);
        posToWordMap.get("N").put("mine", -2.4);
        posToWordMap.get("N").put("saw", -1.7);
        posToWordMap.get("N").put("uses", -2.4);
        posToWordMap.put("PRO", new HashMap<String, Double>());
        posToWordMap.get("PRO").put("I", -1.9);
        posToWordMap.get("PRO").put("many", -1.9);
        posToWordMap.get("PRO").put("me", -1.9);
        posToWordMap.get("PRO").put("mine", -1.9);
        posToWordMap.get("PRO").put("you", -0.8);
        posToWordMap.put("V", new HashMap<String, Double>());
        posToWordMap.get("V").put("color", -2.1);
        posToWordMap.get("V").put("cook", -1.4);
        posToWordMap.get("V").put("eats", -2.1);
        posToWordMap.get("V").put("fish", -2.1);
        posToWordMap.get("V").put("has", -1.4);
        posToWordMap.get("V").put("uses", -2.1);
        posToWordMap.put("MOD", new HashMap<String, Double>());
        posToWordMap.get("MOD").put("can", -0.7);
        posToWordMap.get("MOD").put("will", -0.7);

        //run viterbi on sample sentence and print
        System.out.println(viterbi("will eats the fish"));

    }

    /**
     * Test data from the PSet 5 essentials document -- same as test0, but with the log probability scores
     */
    public void test2() {
        //hard code in the occurrence maps
        posMap = new HashMap<>();
        posMap.put("#", new HashMap<String, Double>());
        posMap.get("#").put("N", -0.3368723);
        posMap.get("#").put("NP", -1.2552661);
        posMap.put("CNJ", new HashMap<String, Double>());
        posMap.get("CNJ").put("N", -1.0996128);
        posMap.get("CNJ").put("NP", -1.0996128);
        posMap.get("CNJ").put("V", -1.0996128);
        posMap.put("N", new HashMap<String, Double>());
        posMap.get("N").put("CNJ", -1.38);
        posMap.get("N").put("V", -0.2876821);
        posMap.put("NP", new HashMap<String, Double>());
        posMap.get("NP").put("V", 0.0);
        posMap.put("V", new HashMap<String, Double>());
        posMap.get("V").put("CNJ", -2.1982251);
        posMap.get("V").put("N", -0.4064656);
        posMap.get("V").put("NP", -1.5050779);

        posToWordMap = new HashMap<>();
        posToWordMap.put("NP", new HashMap<String, Double>());
        posToWordMap.get("NP").put("chase", 0.0);
        posToWordMap.put("CNJ", new HashMap<String, Double>());
        posToWordMap.get("CNJ").put("and", 0.0);
        posToWordMap.put("V", new HashMap<String, Double>());
        posToWordMap.get("V").put("get", -2.1982251);
        posToWordMap.get("V").put("chase", -1.5050779);
        posToWordMap.get("V").put("watch", -0.4064656);
        posToWordMap.put("N", new HashMap<String, Double>());
        posToWordMap.get("N").put("cat", -0.87707);
        posToWordMap.get("N").put("dog", -0.87707);
        posToWordMap.get("N").put("watch", 1.8325815);
        //run viterbi on sample sentence and print it out
        System.out.println("the test sentences is: chase watch dog chase watch");
        System.out.println(viterbi("chase watch dog chase watch"));
    }

    /**
     * Test case using the simple files data
     */
    public void test3() {
        //train the model on the simple files
        posMap = new HashMap<>();
        posToWordMap = new HashMap<>();
        train("inputs/texts/simple-train-tags.txt", "inputs/texts/simple-train-sentences.txt");
        String testSent = "the fast dog is beautiful in the night .";
        System.out.println("The test sentence is: " + testSent);
        System.out.println(viterbi(testSent));

    }

    public static void main(String[] args) {
        POSTagger driver;
        Scanner scan = new Scanner(System.in);
        System.out.println("Welcome to the Viterbi tester!");
        System.out.println("This program will take in an input  and output the most likely part of speech tags given " +
                "its model and training data");
        int trainingCorpus;
        //get the user to decide which training corpus to use
        while(true) {
            System.out.println("Please select which body you would like to train the model on:");
            System.out.println("Enter 1 for the simple, enter 2 for the brown corpus");
            trainingCorpus = scan.nextInt();
            String tagsFileName;
            String sentencesFileName;
            if (trainingCorpus == 1) {
                tagsFileName = "inputs/texts/simple-train-tags.txt";
                sentencesFileName = "inputs/texts/simple-train-sentences.txt";
                driver = new POSTagger(tagsFileName, sentencesFileName);
                break;
            }
            else if (trainingCorpus == 2) {
                tagsFileName = "inputs/texts/brown-train-tags.txt";
                sentencesFileName = "inputs/texts/brown-train-sentences.txt";
                driver = new POSTagger(tagsFileName, sentencesFileName);
                break;
            }
            else {
                System.out.println("Invalid input");
            }
        }

        //ask user if they would like to input their own string or measure the model's success on the test file
        while(true) {
            System.out.println("Enter 1 to input your own string, enter 2 to measure the model's success on the test" +
                    " file, enter 3 to quit");
            int input = scan.nextInt();
            String junk = scan.nextLine(); //read whitespace
            if(input == 1) { //if they choose to input their own string
                System.out.println("Write it here, and make sure to add a space before the punctuation!");
                String inputLine = scan.nextLine();
                System.out.println(driver.viterbi(inputLine));
            }
            else if (input == 2) { //if they choose to test the model's success on the given test files
                List<Integer> results;
                if (trainingCorpus == 1) { //simple corpus
                    results = driver.testModel("inputs/texts/simple-test-tags.txt",
                            "inputs/texts/simple-test-sentences.txt");
                    System.out.println("The model go " + results.get(0) + " tags correct vs " + results.get(1) + " tags wrong.");

                }
                else if (trainingCorpus == 2) { //brown corpus
                    results = driver.testModel("inputs/texts/brown-test-tags.txt",
                            "inputs/texts/brown-test-sentences.txt");
                    System.out.println("The model go " + results.get(0) + " tags correct vs " + results.get(1) + " tags wrong.");
                }
            }
            else if (input == 3) {
                System.out.println("Thank you for playing!");
                break;
            }
            else {
                System.out.println("Invalid input");
            }
        }


        //POSTagger tester = new POSTagger("inputs/texts/simple-train-tags.txt", "inputs/texts/simple-train-sentences.txt");
        //tester.test0();
        //tester.test2();
        //tester.test3();
    }
}
