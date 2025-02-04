package classes;
import interfaces.*;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

public class Home implements ActionListener, IHome{

	private JFrame frame;
	private JLabel iFood;
	private JLabel lLogo;
	private JLabel label1;
	private JLabel label2;
	private JLabel label3;
	private JLabel label4;
	private JLabel label5;
	private JLabel lProfile;
	private JButton findResturant;
	private JMenuBar menuBar;  
	private JMenu users, setting, aboutUs;   
	private JMenuItem faysal, vuban, emon, yeamin,tawhid, profile, logout, changeBGColor;
	private String f="Faysal";
	
	public Home(String user)
	{
		f=user;
		
		frame = new JFrame("Home");
		
		ImageIcon framelogo = new ImageIcon("icons/framelogo.jpg");
		ImageIcon food = new ImageIcon("icons/food.png");
		ImageIcon logoType = new ImageIcon("icons/logotype.png");
		ImageIcon profileImage = new ImageIcon("icons/admin.png");
		ImageIcon resturantLogo = new ImageIcon("icons/resturantlogo.png");
		ImageIcon deliveryLogo = new ImageIcon("icons/delivery.png");
		ImageIcon emojiIcon = new ImageIcon("icons/emoji.png");
		
		faysal = new JMenuItem ("Faysal");
		faysal.addActionListener(this);
		faysal.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		vuban = new JMenuItem ("Vuban");
		vuban.addActionListener(this);
		vuban.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		emon = new JMenuItem ("Emon");
		emon.addActionListener(this);
		emon.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		yeamin = new JMenuItem ("Yeamin");
		yeamin.addActionListener(this);
		yeamin.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		tawhid = new JMenuItem ("Tawhid");
		tawhid.addActionListener(this);
		tawhid.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		profile = new JMenuItem ("Profile");
		profile.addActionListener(this);
		profile.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		logout = new JMenuItem ("Logout");
		logout.addActionListener(this);
		logout.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		changeBGColor = new JMenuItem("Change Background Color");
		changeBGColor.addActionListener(this);
		changeBGColor.setCursor(new Cursor(Cursor.HAND_CURSOR));
		
		
		users = new JMenu("User");
		users.setCursor(new Cursor(Cursor.HAND_CURSOR));
		users.setBorder(new EmptyBorder(0, 10, 0, 10));
		users.setForeground(Color.white);
		users.add(profile);
		users.add(logout);
		
		setting = new JMenu ("Settings");
		setting.setCursor(new Cursor(Cursor.HAND_CURSOR));
		setting.setBorder(new EmptyBorder(0, 10, 0, 10));
		setting.setForeground(Color.white);
		setting.add(changeBGColor);
		
		aboutUs = new JMenu("About Us");
		aboutUs.setCursor(new Cursor(Cursor.HAND_CURSOR));
		aboutUs.setBorder(new EmptyBorder(0, 20, 0, 10));
		aboutUs.setForeground(Color.white);
		aboutUs.add(faysal);
		aboutUs.add(vuban);
		aboutUs.add(yeamin);
		aboutUs.add(emon);
		aboutUs.add(tawhid);
		
		
		menuBar = new JMenuBar();
		menuBar.add(users);
		menuBar.add(setting);
		menuBar.add(aboutUs);
		menuBar.setBackground(Color.black);
		menuBar.setBorder(BorderFactory.createEmptyBorder());
		frame.setJMenuBar(menuBar);
		
		
		lLogo = new JLabel (logoType);
		lLogo.setBounds(575, 210, 160, 120);
		frame.add(lLogo);
		
		
		iFood = new JLabel (food);
		iFood.setBounds(430, 50, 450, 450);
		frame.add(iFood);
		
		
		label1 = new JLabel("Order Tasty &");
		label1.setBounds(30, 30, 300, 150);
		label1.setForeground(Color.white);
		label1.setFont(new Font("Lucida Console",Font.BOLD,28));
		frame.add(label1);
		
		
		label2 = new JLabel("Fresh Food");
		label2.setBounds(30, 70, 300, 150);
		label2.setForeground(Color.white);
		label2.setFont(new Font("Lucida Console",Font.BOLD,28));
		frame.add(label2);
		
		
		label3 = new JLabel("anytime!");
		label3.setBounds(30, 110, 300, 150);
		label3.setForeground(Color.red);
		label3.setFont(new Font("Lucida Console",Font.BOLD,28));
		frame.add(label3);
		
		
		label4 = new JLabel(deliveryLogo);
		label4.setBounds(20, 300, 200, 200);
		frame.add(label4);
		
		
		label5 = new JLabel("EASY WAY TO ORDER YOUR FOOD");
		label5.setIcon(emojiIcon);
		label5.setBounds(30, 10, 250, 100);
		label5.setForeground(Color.white);
		label5.setFont(new Font("Lucida Console",Font.PLAIN,10));
		frame.add(label5);
		
		
		lProfile = new JLabel (user);
		lProfile.setBounds(800, 0, 80, 23);
		lProfile.setForeground(Color.black);
		lProfile.setBackground(Color.white);
		lProfile.setHorizontalAlignment(JLabel.CENTER);
		lProfile.setOpaque(true);
		lProfile.setIcon(profileImage);
		frame.add(lProfile);
		
		
		findResturant= new JButton("Find Resturant");
		findResturant.setBounds(240, 250, 150, 30);
		findResturant.setFocusable(false);
		findResturant.setBackground(Color.black);
		findResturant.setForeground(Color.white);
		findResturant.setBorder(BorderFactory.createDashedBorder(Color.red, 1.9F, 5, 1, true));
		//findResturant.setBorder(BorderFactory.createEtchedBorder(0, Color.green, Color.red));
		findResturant.setIcon(resturantLogo);
		findResturant.setCursor(new Cursor(Cursor.HAND_CURSOR));
		findResturant.addActionListener(this);
		frame.add(findResturant);
		
		
		frame.setIconImage(framelogo.getImage());
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(900,600);
		frame.getContentPane().setBackground(Color.BLACK);
		frame.setLayout (null);
		frame.setResizable(false);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}

	public static void main (String [] args)
	{
		new Home("Faysal");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==logout)
		{
			frame.setVisible(false);
			new Login();
		}
		else if(e.getSource()==changeBGColor)
		{
				JColorChooser colorChooser = new JColorChooser();

				Color color = JColorChooser.showDialog(null, "Pick Your Background Color", Color.black);
				
				frame.getContentPane().setBackground(color);
				
				menuBar.setBackground(color);
		}
		else if(e.getSource()==findResturant)
		{
			frame.setVisible(false);
			new ResturantName(f);
		}
		else if(e.getSource()==profile)
		{
			frame.setVisible(false);
			new UProfile(f);
		}
		
		
		
		//need to work
		else if(e.getSource()==faysal)
		{
			frame.setVisible(false);
			new FaysalProfile(f);
		}
		else if(e.getSource()==vuban)
		{
			new VubanProfile(f);
			frame.setVisible(false);
		}
		else if(e.getSource()==emon)
		{
			new EmonProfile(f);
			frame.setVisible(false);
		}
		else if(e.getSource()==yeamin)
		{
			new YeaminProfile(f);
			frame.setVisible(false);
		}
		else if(e.getSource()==tawhid)
		{
			new TawhidProfile(f);
			frame.setVisible(false);
		}
		
	}
}


