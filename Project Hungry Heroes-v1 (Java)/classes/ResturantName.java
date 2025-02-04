package classes;
import interfaces.*;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.border.EmptyBorder;

public class ResturantName implements ActionListener, MouseListener, IResturantName{
	
	private JFrame frame;
	
	private JLabel sultansDine;
	private JLabel pizzaHut;
	private JLabel in_N_Out;
	private JLabel tacoBell;
	private JLabel twinPeaks;
	private JLabel treat;
	
	private JButton goBack;
	
	private JMenuBar menuBar;  
	private JMenu users, setting;
	private JMenuItem profile, logout, changeBGColor;
	
	private String f ="Faysal";
	
	public ResturantName (String user)
	{
		f=user;
		frame = new JFrame("Available Resturant");
		
		ImageIcon framelogo = new ImageIcon("icons/framelogo.jpg");
		Image image1 = new ImageIcon("icons/SultansDine.jpg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon sultansDineLogo = new ImageIcon(image1);
		sultansDine = new JLabel("Sultans Dine");
		sultansDine.setIcon(sultansDineLogo);
		sultansDine.setForeground(Color.CYAN);
		sultansDine.setVerticalTextPosition(JLabel.BOTTOM);
		sultansDine.setHorizontalTextPosition(JLabel.CENTER);
		sultansDine.setCursor(new Cursor(Cursor.HAND_CURSOR));
		sultansDine.setFont(new Font("Lucida Console",Font.PLAIN,20));
		sultansDine.addMouseListener(this);
		frame.add(sultansDine);
		
		
		Image image2 = new ImageIcon("icons/PizzaHut.jpg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon pizzaHutLogo = new ImageIcon(image2);
		pizzaHut = new JLabel("Pizza Hut");
		pizzaHut.setIcon(pizzaHutLogo);
		pizzaHut.setForeground(Color.CYAN);
		pizzaHut.setVerticalTextPosition(JLabel.BOTTOM);
		pizzaHut.setHorizontalTextPosition(JLabel.CENTER);
		pizzaHut.setCursor(new Cursor(Cursor.HAND_CURSOR));
		pizzaHut.setFont(new Font("Lucida Console",Font.PLAIN,20));
		pizzaHut.addMouseListener(this);
		frame.add(pizzaHut);
		
		
		Image image3 = new ImageIcon("icons/In-N-Out-Burger.jpg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon in_N_OutLogo = new ImageIcon(image3);
		in_N_Out = new JLabel("IN-N-OUT BURGER");
		in_N_Out.setIcon(in_N_OutLogo);
		in_N_Out.setForeground(Color.CYAN);
		in_N_Out.setVerticalTextPosition(JLabel.BOTTOM);
		in_N_Out.setHorizontalTextPosition(JLabel.CENTER);
		in_N_Out.setCursor(new Cursor(Cursor.HAND_CURSOR));
		in_N_Out.setFont(new Font("Lucida Console",Font.PLAIN,20));
		in_N_Out.addMouseListener(this);
		frame.add(in_N_Out);
		
		
		Image image4 = new ImageIcon("icons/TacoBell.jpeg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon tacoBellLogo = new ImageIcon(image4);
		tacoBell = new JLabel("Taco Bell");
		tacoBell.setIcon(tacoBellLogo);
		tacoBell.setForeground(Color.CYAN);
		tacoBell.setVerticalTextPosition(JLabel.BOTTOM);
		tacoBell.setHorizontalTextPosition(JLabel.CENTER);
		tacoBell.setCursor(new Cursor(Cursor.HAND_CURSOR));
		tacoBell.setFont(new Font("Lucida Console",Font.PLAIN,20));
		tacoBell.addMouseListener(this);
		frame.add(tacoBell);
		
		
		Image image5 = new ImageIcon("icons/TwinPeaks.jpg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon twinPeaksLogo = new ImageIcon(image5);
		twinPeaks = new JLabel("Twin Peaks");
		twinPeaks.setIcon(twinPeaksLogo);
		twinPeaks.setForeground(Color.CYAN);
		twinPeaks.setVerticalTextPosition(JLabel.BOTTOM);
		twinPeaks.setHorizontalTextPosition(JLabel.CENTER);
		twinPeaks.setCursor(new Cursor(Cursor.HAND_CURSOR));
		twinPeaks.setFont(new Font("Lucida Console",Font.PLAIN,20));
		twinPeaks.addMouseListener(this);
		frame.add(twinPeaks);
		
		
		Image image6 = new ImageIcon("icons/treat.jpg").getImage().getScaledInstance(240, 180, Image.SCALE_DEFAULT);
		ImageIcon treatLogo = new ImageIcon(image6);
		treat = new JLabel("Treat");
		treat.setIcon(treatLogo);
		treat.setForeground(Color.CYAN);
		treat.setVerticalTextPosition(JLabel.BOTTOM);
		treat.setHorizontalTextPosition(JLabel.CENTER);
		treat.setCursor(new Cursor(Cursor.HAND_CURSOR));
		treat.setFont(new Font("Lucida Console",Font.PLAIN,20));
		treat.addMouseListener(this);
		frame.add(treat);


		profile = new JMenuItem ("Profile");
		profile.addActionListener(this);
		profile.setCursor(new Cursor(Cursor.HAND_CURSOR));
		profile.setVisible(false);
		
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
		
		
		goBack= new JButton("Previous page");
		goBack.setFocusable(false);
		goBack.setBackground(Color.black);
		goBack.setForeground(Color.white);
		goBack.setCursor(new Cursor(Cursor.HAND_CURSOR));
		goBack.setBorder(BorderFactory.createDashedBorder(Color.red));
		goBack.addActionListener(this);
		
		
		menuBar = new JMenuBar();
		menuBar.add(users);
		menuBar.add(setting);
		menuBar.add(goBack);
		menuBar.setBackground(Color.black);
		menuBar.setBorder(BorderFactory.createEmptyBorder());
		frame.setJMenuBar(menuBar);
	
		
		frame.setIconImage(framelogo.getImage());
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(900,600);
		frame.getContentPane().setBackground(Color.BLACK);
		frame.setLayout (new GridLayout(2,3));
		frame.setResizable(false);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
	public static void main (String [] args)
	{
		new ResturantName("Faysal");
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
				
				goBack.setBackground(color);
		}
		
		else if (e.getSource()==goBack)
		{
			new Home (f);
			frame.setVisible(false);
		}
		
	}
	
	@Override
	public void mouseClicked(MouseEvent e) {
		if(e.getSource()==pizzaHut)
		{
			frame.setVisible(false);
			new PizzaHut(f);
		}
		else if(e.getSource()==sultansDine)
		{
			frame.setVisible(false);
			new SultansDine(f);
		}
		else if(e.getSource()==tacoBell)
		{
			frame.setVisible(false);
			new TacoBell(f);
		}
		else if(e.getSource()==in_N_Out)
		{
			frame.setVisible(false);
			new In_N_OutBurger(f);
		}
		else if(e.getSource()==twinPeaks)
		{
			frame.setVisible(false);
			new TwinPeaks(f);
		}
		else if(e.getSource()==treat)
		{
			frame.setVisible(false);
			new Treat(f);
		}
	
	}

	@Override
	public void mousePressed(MouseEvent e) {
		
		
	}

	@Override
	public void mouseReleased(MouseEvent e) {
	
		
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		if(e.getSource()==sultansDine)
			{
			sultansDine.setForeground(Color.red);
			}
		
		else if(e.getSource()==pizzaHut)
			{
			pizzaHut.setForeground(Color.red);
			}
		
		else if(e.getSource()==in_N_Out)
			{
			in_N_Out.setForeground(Color.red);
			}
		
		else if(e.getSource()==tacoBell)
			{
			tacoBell.setForeground(Color.red);
			}
		
		else if(e.getSource()==twinPeaks)
			{
			twinPeaks.setForeground(Color.red);
			}
		else if(e.getSource()==treat)
			{
			treat.setForeground(Color.red);
			}
		
	}
			
	@Override
	public void mouseExited(MouseEvent e) {
		sultansDine.setForeground(Color.cyan);
		sultansDine.setForeground(Color.cyan);
		pizzaHut.setForeground(Color.cyan);
		in_N_Out.setForeground(Color.cyan);
		tacoBell.setForeground(Color.cyan);
		twinPeaks.setForeground(Color.cyan);
		treat.setForeground(Color.cyan);
		
	}

}
