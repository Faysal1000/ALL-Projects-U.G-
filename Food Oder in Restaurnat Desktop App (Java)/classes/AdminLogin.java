package classes;
import interfaces.*;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.Image;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPasswordField;
import javax.swing.JTextField;

public class AdminLogin implements MouseListener, IAdminLogin{

	private JFrame frame;
	private JLabel label1;
	private JLabel label2;
	private JLabel userName;
	private JLabel userPass;
	private JTextField tfUName;
	private JPasswordField pfUPass;
	private User u1,u2;
	private User users[];
	
	public AdminLogin()
	{
	
		frame = new JFrame("Admin Login");
		
		ImageIcon framelogo = new ImageIcon("icons/framelogo.jpg");
		ImageIcon userNameImage = new ImageIcon("icons/username.png");
		ImageIcon UserPassImage = new ImageIcon("icons/userpass.png");
		
		Image background = new ImageIcon("icons/adminLogin.jpg").getImage().getScaledInstance(550, 550, Image.SCALE_AREA_AVERAGING);
		frame.setContentPane(new ImagePanel(background));
		
		
		u1 = new User("Faysal","21", "1234");
		u2 = new User("admin","21", "admin");
		users = new User[4];
		users[0] = u1;
		users[1] = u2;
		
		
		label1 = new JLabel("Admin LogIn");
		label1.setForeground(Color.white);
		label1.setFont(new Font("Imprint MT Shadow",Font.PLAIN,25));
		label1.setBounds(200, 120, 200, 80);
		frame.add(label1);
		
		
		label2 = new JLabel("Admin In");
		label2.setForeground(Color.white);
		label2.setFont(new Font("Segoe UI Black",Font.PLAIN,18));
		label2.setBounds(230, 440, 100, 80);
		label2.setCursor(new Cursor(Cursor.HAND_CURSOR));
		label2.addMouseListener(this);
		frame.add(label2);
		
		
		userName = new JLabel("Admin name: ");
		userName.setBounds(110, 240, 120, 30);
		userName.setFont(new Font("Calibri",Font.BOLD,16));
		userName.setIcon(userNameImage);
		userName.setForeground(Color.decode("#FFFF00"));
		frame.add(userName);
		
		
		userPass = new JLabel("Password: ");
		userPass.setBounds(130, 280, 100, 30);
		userPass.setFont(new Font("Calibri",Font.BOLD,16));
		userPass.setIcon(UserPassImage);
		userPass.setForeground(Color.decode("#FFFF00"));
		frame.add(userPass);
		
		
		tfUName = new JTextField();
		tfUName.setBounds(230, 240, 140, 23);
		tfUName.setBorder(BorderFactory.createLineBorder(Color.GREEN, 1));
		tfUName.setToolTipText("Enter your name here");
		frame.add(tfUName);
		
		
		pfUPass = new JPasswordField();
		pfUPass.setBounds(230, 280, 140, 23);
		pfUPass.setBorder(BorderFactory.createLineBorder(Color.GREEN, 1));
		pfUPass.setEchoChar('*');
		pfUPass.setToolTipText("Enter your password here");
		frame.add(pfUPass);
		

		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(550, 585);
		frame.getContentPane().setBackground(Color.black);
		frame.setIconImage(framelogo.getImage());
		frame.setResizable(false);
		frame.setLayout (null);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
		
	}
	 public User searchUser(String username)
	    {
	    	User a = null;
	    	
	    	for(int i=0; i<users.length; i++)
	    	{
	    		if(users[i]!=null)
	    		{
	    			if(users[i].getName().equals(username))
	        		{
	        			a = users[i];
	        			break;
	        		}
	    		}
	    	}
	    	return a;
	    }
	
	public static void main(String[] args) {
		
			new AdminLogin();
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		if (e.getSource()==label2)
		{
			String user = tfUName.getText();
			String pass = pfUPass.getText();
			int flag=0;
			if(user.isEmpty()==false && pass.isEmpty()==false)
				{
				for(int i = 0; i<users.length; i++)
				{
					if(users[i]!=null)
					{
						if(user.equals(users[i].getName()) && pass.equals(users[i].getPassword()))
						{
							flag = 1;
							break;
						}
					}
				}
				if (flag == 1)
				{
					User x= this.searchUser(user);
					new AdminHome(x);
					frame.setVisible(false);
				}
				else
				{
				JOptionPane.showMessageDialog(null, "Invalid Admin name or password!");
				}
			}
			else
			{
			JOptionPane.showMessageDialog(null,"Admin Name & Password field can't be empty!");
			}
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
		if (e.getSource()==label2)
		{
			label2.setForeground(Color.green);
		}
		
	}

	@Override
	public void mouseExited(MouseEvent e) {
		if (e.getSource()==label2)
		{
			label2.setForeground(Color.white);
		}
		
	}

}
