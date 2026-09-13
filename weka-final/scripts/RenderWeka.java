import java.nio.file.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;
import weka.gui.treevisualizer.*;
public class RenderWeka {
 public static void main(String[] a)throws Exception{
  Path p=Path.of(a[0]);
  for(String key:new String[]{"sgpa","questionnaire","main"}){
   String dot=Files.readString(p.resolve(key+"_tree.dot"));
   SwingUtilities.invokeAndWait(()->{try{
    TreeVisualizer v=new WekaEvidence.Viewer(dot);v.setSize(1800,950);
    JRadioButtonMenuItem size=new JRadioButtonMenuItem();size.setActionCommand("Size 24");v.itemStateChanged(new java.awt.event.ItemEvent(size,java.awt.event.ItemEvent.ITEM_STATE_CHANGED,size,java.awt.event.ItemEvent.SELECTED));
    BufferedImage img=new BufferedImage(1800,950,BufferedImage.TYPE_INT_RGB);Graphics2D g=img.createGraphics();
    g.setColor(Color.WHITE);g.fillRect(0,0,1800,950);v.printAll(g);v.fitToScreen();g.setColor(Color.WHITE);g.fillRect(0,0,1800,950);v.printAll(g);v.fitToScreen();g.setColor(Color.WHITE);g.fillRect(0,0,1800,950);v.printAll(g);g.dispose();
    ImageIO.write(img,"png",p.resolve(key+"_weka_viewer.png").toFile());
    String log=Files.readString(p.resolve(key+"_J48_holdout.txt"));
    String excerpt="WEKA 3.8.7    "+key+"    J48 -C 0.25 -M 40\n\n"+log.substring(log.indexOf("=== TEST ==="),log.indexOf("=== ZeroR TEST BASELINE ==="));
    JTextArea t=new JTextArea(excerpt);t.setFont(new Font("Monospaced",Font.PLAIN,22));t.setForeground(Color.BLACK);t.setBackground(Color.WHITE);t.setBorder(BorderFactory.createEmptyBorder(25,25,25,25));t.setSize(1800,1050);
    BufferedImage r=new BufferedImage(1800,1050,BufferedImage.TYPE_INT_RGB);Graphics2D rg=r.createGraphics();t.printAll(rg);rg.dispose();ImageIO.write(r,"png",p.resolve(key+"_weka_output.png").toFile());
   }catch(Exception e){throw new RuntimeException(e);}});
  }
  System.exit(0);
 }
}
