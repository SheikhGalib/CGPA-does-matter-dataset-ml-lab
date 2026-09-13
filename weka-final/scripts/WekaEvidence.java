import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.functions.*;
import weka.classifiers.bayes.*;
import weka.classifiers.rules.*;
import weka.gui.treevisualizer.*;
import java.nio.file.*;
import java.util.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;

public class WekaEvidence {
 static Path out;
 static Instances load(String name)throws Exception {Instances d=DataSource.read(out.resolve(name+".arff").toString());d.setClassIndex(d.numAttributes()-1);return d;}
 static Classifier model(String n)throws Exception {return switch(n){case "J48"->new J48();case "RandomForest"->new RandomForest();case "SMO"->new SMO();case "Logistic"->new Logistic();case "NaiveBayes"->new NaiveBayes();default->new ZeroR();};}
 static String report(Evaluation e)throws Exception{return e.toSummaryString()+"\n"+e.toClassDetailsString()+"\n"+e.toMatrixString();}
 static void save(String name,String text)throws Exception{Files.writeString(out.resolve(name),text);}
 static class Viewer extends TreeVisualizer {
  Viewer(String graph){super(null,graph,new PlaceNode2());m_BackgroundColor=Color.WHITE;m_NodeColor=new Color(230,236,240);m_FontColor=Color.BLACK;m_LineColor=Color.DARK_GRAY;}
 }
 public static void main(String[] args)throws Exception {
  out=Path.of(args[0]);
  for(String key:new String[]{"sgpa","questionnaire","main"}){
   Instances all=load(key+"_all"),train=load(key+"_train"),test=load(key+"_test");
   for(String name:new String[]{"J48","RandomForest","SMO","Logistic","NaiveBayes","ZeroR"}){
    Classifier c=model(name);Evaluation e=new Evaluation(all);e.crossValidateModel(c,all,10,new Random(1));
    String settings=Utils.joinOptions(((OptionHandler)c).getOptions());
    save(key+"_"+name+"_cv.txt","WEKA "+Version.VERSION+"\nTarget experiment: "+key+"\nModel: "+name+"\nOptions: "+settings+"\nEvaluation: stratified 10-fold CV, seed 1\nDataset N: "+all.numInstances()+"\n"+report(e));
    System.out.println(key+" "+name+" CV "+e.pctCorrect());
   }
   J48 tree=new J48();tree.setOptions(new String[]{"-C","0.25","-M","40"});tree.buildClassifier(train);
   Evaluation tr=new Evaluation(train);tr.evaluateModel(tree,train);Evaluation te=new Evaluation(train);te.evaluateModel(tree,test);
   ZeroR zr=new ZeroR();zr.buildClassifier(train);Evaluation ze=new Evaluation(train);ze.evaluateModel(zr,test);
   save(key+"_J48_holdout.txt","WEKA "+Version.VERSION+"\nTarget experiment: "+key+"\nModel: WEKA J48 Decision Tree\nOptions: "+Utils.joinOptions(tree.getOptions())+"\nDataset: "+train.numInstances()+" train, "+test.numInstances()+" test\n"+tree.toString()+"\n=== TRAIN ===\n"+report(tr)+"\n=== TEST ===\n"+report(te)+"\n=== ZeroR TEST BASELINE ===\n"+report(ze));
   save(key+"_tree.txt",tree.toString());save(key+"_tree.dot",tree.graph());SerializationHelper.write(out.resolve(key+"_J48.model").toString(),tree);
   String graph=tree.graph();
   SwingUtilities.invokeAndWait(()->{try{
    Viewer v=new Viewer(graph);v.setSize(1800,950);v.doLayout();v.fitToScreen();
    BufferedImage img=new BufferedImage(1800,950,BufferedImage.TYPE_INT_RGB);Graphics2D g=img.createGraphics();g.setColor(Color.WHITE);g.fillRect(0,0,1800,950);v.printAll(g);g.dispose();ImageIO.write(img,"png",out.resolve(key+"_weka_viewer.png").toFile());
   }catch(Exception ex){throw new RuntimeException(ex);}});
   System.out.println(key+" J48 tree nodes "+tree.measureTreeSize()+" test "+te.pctCorrect());
  }
  System.exit(0);
 }
}
