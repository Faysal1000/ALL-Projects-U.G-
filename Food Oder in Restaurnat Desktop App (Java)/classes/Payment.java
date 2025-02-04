package classes;
import interfaces.*;
import java.awt.Color;
import java.awt.Font;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.InputMismatchException;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.border.EtchedBorder;

public class Payment implements ActionListener, IPayment{
	
	private JFrame frame;
	private JLabel cyp;
	private JLabel wao;
	private JLabel wao2;
	private JLabel pfuycd;
	private JLabel noc;
	private JLabel cn;
	private JLabel vn;
	private JLabel cvvc;
	private JLabel mi;
	
	private JTextField tfUName;
	private JTextField tfcn;
	private JTextField tfvo;
	private JPasswordField pfcvvc;
	
	private JButton cpB;
	private JButton exitButton;
	private JButton previous;
	
	private PizzaHut x;
	
	private double z;
	
	private String resturantName;
	private String f;
	
	public Payment(double amount, String name, String user)
	{
		f= user;
		
		resturantName= name;
		
		z=amount;
		frame = new JFrame("Payment");
		
		ImageIcon framelogo = new ImageIcon("icons/framelogo.jpg");
		
		Image image1 = new ImageIcon("icons/mainIcon.png").getImage().getScaledInstance(300, 450, Image.SCALE_DEFAULT);
		ImageIcon mainIcon = new ImageIcon(image1);
		mi = new JLabel(mainIcon);
		mi.setBounds(20, 10, 300, 450);
		frame.add(mi);
		
		
		cyp = new JLabel("Complete Your Payment");
		cyp.setForeground(Color.BLACK);
		cyp.setFont(new Font("Segoe UI Black", Font.PLAIN, 34));
		cyp.setBounds(420, 20, 500, 50);
		frame.add(cyp);
		
		
		ImageIcon waoIcon = new ImageIcon("icons/wao1.png");
		wao = new JLabel("We Accept Only");
		wao.setForeground(Color.BLACK);
		wao.setIcon(waoIcon);
		wao.setHorizontalTextPosition(JLabel.LEFT);
		wao.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 25));
		wao.setBounds(420, 80, 500, 50);
		frame.add(wao);
		
		
		ImageIcon waoIcon2 = new ImageIcon("icons/wao2.png");
		wao2= new JLabel(waoIcon2);
		wao2.setBounds(480, 80, 500, 50);
		frame.add(wao2);
		
		
		pfuycd = new JLabel("Please Fillup Your Card Details Below---");
		pfuycd.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 20));
		pfuycd.setBounds(420, 130, 400, 50);
		frame.add(pfuycd);
		
		
		noc = new JLabel("Name On Card :");
		noc.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 18));
		noc.setBounds(420, 170, 150, 50);
		frame.add(noc);
		
		
		cn = new JLabel("Card Number :");
		cn.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 18));
		cn.setBounds(420, 210, 500, 50);
		frame.add(cn);
		
		
		vn = new JLabel("Valid On: ");
		vn.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 18));
		vn.setBounds(420, 250, 500, 50);
		frame.add(vn);
		
		cvvc = new JLabel("CVV Code: ");
		cvvc.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 18));
		cvvc.setBounds(610, 250, 500, 50);
		frame.add(cvvc);
		
		
		tfUName = new JTextField();
		tfUName.setBounds(570, 185, 240, 23);
		tfUName.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
		tfUName.setToolTipText("Your name?");
		tfUName.setVisible(true);
		frame.add(tfUName);
		
		
		tfcn = new JTextField();
		tfcn.setBounds(570, 225, 240, 23);
		tfcn.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
		tfcn.setToolTipText("Your card number?");
		tfcn.setVisible(true);
		frame.add(tfcn);
		
		
		tfvo = new JTextField();
		tfvo.setBounds(505, 265, 100, 23);
		tfvo.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
		tfvo.setToolTipText("Your card validity date?");
		tfvo.setVisible(true);
		frame.add(tfvo);
		
		
		pfcvvc = new JPasswordField();
		pfcvvc.setBounds(710, 265, 100, 23);
		pfcvvc.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
		pfcvvc.setToolTipText("Your CVV code?");
		pfcvvc.setEchoChar('*');
		pfcvvc.setVisible(true);
		frame.add(pfcvvc);
		
		
		cpB= new JButton("Confirm Payment");
		cpB.setBounds(575, 350, 150, 30);
		cpB.setFocusable(false);
		cpB.setBackground(Color.green);
		cpB.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 14));
		cpB.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.RAISED));
		cpB.addActionListener(this);
		cpB.setVisible(true);
		frame.add(cpB);
		
		
		exitButton = new JButton("Exit");
		exitButton.setBounds(700, 410, 100, 30);
		exitButton.setFocusable(false);
		exitButton.setForeground(Color.WHITE);
		exitButton.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 15));
		exitButton.setBackground(Color.DARK_GRAY);
		exitButton.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.RAISED));
		exitButton.addActionListener(this);
		frame.add(exitButton);
		
		
		previous = new JButton ("Previous");
		previous.setBounds(500, 410, 100, 30);
		previous.setFocusable(false);
		previous.setBackground(Color.DARK_GRAY);
		previous.setForeground(Color.WHITE);
		previous.setFont(new Font("Arial Rounded MT Bold", Font.PLAIN, 14));
		previous.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.RAISED));
		previous.addActionListener(this);
		previous.setVisible(true);
		frame.add(previous);
		
		
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(1000, 550);
		frame.setIconImage(framelogo.getImage());
		frame.getContentPane().setBackground(Color.decode("#F2F2F2"));
		frame.setResizable(false);
		frame.setLayout (null);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}


	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==previous)
		{
			if(resturantName=="PizzaHut")
			{
				frame.setVisible(false);
				new PizzaHut(f);
			}
			else if(resturantName=="SultansDine")
			{
				frame.setVisible(false);
				new SultansDine(f);
			}
			else if(resturantName=="TacoBell")
			{
				frame.setVisible(false);
				new TacoBell(f);
			}
			else if(resturantName=="TwinPeaks")
			{
				frame.setVisible(false);
				new TwinPeaks(f);
			}
			else if(resturantName=="In_N_OutBurger")
			{
				frame.setVisible(false);
				new In_N_OutBurger(f);
			}
			else if(resturantName=="Treat")
			{
				frame.setVisible(false);
				new Treat(f);
			}
			
			
		}
		else if(e.getSource()==exitButton)
		{
			int a = JOptionPane.showConfirmDialog(null, "Do you want to exit?", "Exit Confirmation", 0);
			if(a==0)
			{
				System.exit(0);
			}
		}
		
		else if(e.getSource()==cpB)
		{
			String a = tfUName.getText();
			String b = tfcn.getText();
			String c = tfvo.getText();
			String d = pfcvvc.getText();
			if (a.isEmpty()|| b.isEmpty() || c.isEmpty() || d.isEmpty())
			{
				JOptionPane.showInternalMessageDialog(null, "Please fill up all the necessary details");
			}
			else
			{
				try 
				{
					long m = Long.parseLong(b);
					long n = Long.parseLong(d);
					
					if(b.length()>16)
					{
						JOptionPane.showMessageDialog(null, "Card number should't have more than 16 digit ", "Alert!", 0);
					}
					
					else if (d.length()!=3)
					{
						JOptionPane.showMessageDialog(null, "CVV code number should have 3 digit ", "Alert!", 0);
					}
					
					else if(b.length()<=16 && d.length()==3)
					{
						frame.setVisible(false);
						new PaymentSuccess(z,f);
					}
					try
					{
						File newfile = new File("data\\Last2.txt"); 
						newfile.createNewFile();
						FileWriter writeFile = new FileWriter("data\\Last2.txt");
						
						writeFile.write("Name on Card : "+a+".\nCard Number :"+b+
								".\nValid On : "+c+".\nCVV Code : "+d+".");
						writeFile.close();
						
					}
					
					catch(IOException io)
					{
						JOptionPane.showMessageDialog(null, "An error Occured while writting file");
						io.printStackTrace();
					}
	
				}
				
	
				catch(InputMismatchException a1)
				{
					JOptionPane.showMessageDialog(null, "Please enter valid card no or code!", "Alert!", 0);
				}
				catch(NumberFormatException a2)
				{
					JOptionPane.showMessageDialog(null, "Please enter valid information!", "Alert!", 0);
				}
				
			}
		}
		
	}

}
