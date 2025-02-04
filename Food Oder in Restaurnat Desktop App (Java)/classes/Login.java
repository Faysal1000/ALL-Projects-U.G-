package classes;
import interfaces.*;
import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import javax.swing.*;

public class Login implements ActionListener,ILogin{

	private JFrame frame;
	private JLabel title;
	private JLabel shortNote;
	private JLabel loginAlert;
	private JButton loginButton1;
	private JButton loginButton2;
	private JButton signUpButton1;
	private JButton exitButton;
	private JButton adminButton;
	private JButton previous0;
	private JLabel userName;
	private JLabel userPass;
	private JLabel confirmUserPass;
	private JTextField tfUName;
	private JPasswordField pfUPass;
	private JPasswordField cpfUPass;
	
	public Login()
	{
		frame = new JFrame("Hungry Heroes");
		
		ImageIcon framelogo = new ImageIcon("icons/framelogo.jpg");
		ImageIcon loginImage = new ImageIcon("icons/login.png");
		ImageIcon signupImage = new ImageIcon("icons/signup.png");
		ImageIcon adminImage = new ImageIcon("icons/admin.png");
		ImageIcon userNameImage = new ImageIcon("icons/username.png");
		ImageIcon UserPassImage = new ImageIcon("icons/userpass.png");
		ImageIcon previousImage = new ImageIcon("icons/previous.png");
		
		Image background = new ImageIcon("icons/mainBackground.jpg").getImage();
		frame.setContentPane(new ImagePanel(background));
		
		title = new JLabel("Hungry Heroes");
		title.setForeground(Color.green);
		title.setFont(new Font("Mistral",Font.PLAIN,50));
		title.setBounds(270, 0, 300, 75);
		frame.add(title);
		
		
		shortNote = new JLabel("-Order Tasty & Fresh FoodAnytime");
		shortNote.setForeground(Color.orange);
		shortNote.setFont(new Font("Harlow Solid Italic",Font.PLAIN,17));
		shortNote.setBounds(330, 80, 380, 60);
		frame.add(shortNote);
		
		loginAlert = new JLabel ("Please Login to Continue....");
		loginAlert.setForeground(Color.LIGHT_GRAY);
		loginAlert.setFont(new Font("Imprint MT Shadow",Font.PLAIN,17));
		loginAlert.setBounds(300, 240, 380, 60);
		frame.add(loginAlert);
		
		
		loginButton1= new JButton("Login");
		loginButton1.setBounds(300, 380, 80, 30);
		loginButton1.setFocusable(false);
		loginButton1.setBackground(Color.LIGHT_GRAY);
		loginButton1.setBorder(BorderFactory.createRaisedSoftBevelBorder());
		loginButton1.setIcon(loginImage);
		loginButton1.setCursor(new Cursor(Cursor.HAND_CURSOR));
		loginButton1.addActionListener(this);
		frame.add(loginButton1);
		
		
		signUpButton1 = new JButton("Signup");
		signUpButton1.setBounds(420, 380, 80, 30);
		signUpButton1.setFocusable(false);
		signUpButton1.setBackground(Color.LIGHT_GRAY);
		signUpButton1.setBorder(BorderFactory.createRaisedSoftBevelBorder());
		signUpButton1.setIcon(signupImage);
		signUpButton1.setCursor(new Cursor(Cursor.HAND_CURSOR));
		signUpButton1.addActionListener(this);
		frame.add(signUpButton1);
		
		
		exitButton = new JButton("Exit");
		exitButton.setBounds(360, 440, 80, 30);
		exitButton.setFocusable(false);
		exitButton.setBackground(Color.LIGHT_GRAY);
		exitButton.setCursor(new Cursor(Cursor.HAND_CURSOR));
		exitButton.setBorder(BorderFactory.createRaisedSoftBevelBorder());
		exitButton.addActionListener(this);
		frame.add(exitButton);
		
		
		userName = new JLabel("Username: ");
		userName.setBounds(240, 180, 100, 30);
		userName.setFont(new Font("Calibri",Font.BOLD,16));
		userName.setIcon(userNameImage);
		userName.setForeground(Color.decode("#FFFF00"));
		userName.setVisible(false);
		frame.add(userName);
		
		
		userPass = new JLabel("Password: ");
		userPass.setBounds(240, 220, 100, 30);
		userPass.setFont(new Font("Calibri",Font.BOLD,16));
		userPass.setIcon(UserPassImage);
		userPass.setForeground(Color.decode("#FFFF00"));
		userPass.setVisible(false);
		frame.add(userPass);
		
		
		confirmUserPass = new JLabel("Confirm Password: ");
		confirmUserPass.setBounds(180, 260, 180, 30);
		confirmUserPass.setFont(new Font("Calibri",Font.BOLD,16));
		confirmUserPass.setForeground(Color.decode("#FFFF00"));
		confirmUserPass.setIcon(UserPassImage);
		confirmUserPass.setVisible(false);
		frame.add(confirmUserPass);
		
		
		tfUName = new JTextField();
		tfUName.setBounds(350, 180, 140, 23);
		tfUName.setBorder(BorderFactory.createLineBorder(Color.GREEN, 1));
		tfUName.setToolTipText("Enter your name here");
		tfUName.setVisible(false);
		frame.add(tfUName);
		
		
		pfUPass = new JPasswordField();
		pfUPass.setBounds(350, 220, 140, 23);
		pfUPass.setBorder(BorderFactory.createLineBorder(Color.GREEN, 1));
		pfUPass.setEchoChar('*');
		pfUPass.setToolTipText("Enter your password here");
		pfUPass.setVisible(false);
		frame.add(pfUPass);
		
		
		cpfUPass = new JPasswordField();
		cpfUPass.setBounds(350, 260, 140, 23);
		cpfUPass.setBorder(BorderFactory.createLineBorder(Color.GREEN, 1));
		cpfUPass.setEchoChar('*');
		cpfUPass.setToolTipText("Re-write your password here");
		cpfUPass.setVisible(false);
		frame.add(cpfUPass);
	
		
		loginButton2= new JButton("Login");
		loginButton2.setBounds(300, 380, 80, 30);
		loginButton2.setFocusable(false);
		loginButton2.setBackground(Color.LIGHT_GRAY);
		loginButton2.setBorder(BorderFactory.createRaisedSoftBevelBorder());
		loginButton2.setIcon(loginImage);
		loginButton2.setCursor(new Cursor(Cursor.HAND_CURSOR));
		loginButton2.addActionListener(this);
		loginButton2.setVisible(false);
		frame.add(loginButton2);
		
		
		previous0 = new JButton ("Previous");
		previous0.setBounds(420, 380, 80, 30);
		previous0.setFocusable(false);
		previous0.setBackground(Color.LIGHT_GRAY);
		previous0.setBorder(BorderFactory.createRaisedSoftBevelBorder());
		previous0.addActionListener(this);
		previous0.setCursor(new Cursor(Cursor.HAND_CURSOR));
		previous0.setIcon(previousImage);
		previous0.setVisible(false);
		frame.add(previous0);
		
		
		adminButton = new JButton("Admin");
		adminButton.setBounds(715, 0, 70, 20);
		adminButton.setFocusable(false);
		adminButton.setBackground(Color.red);
		adminButton.setBorder(BorderFactory.createLoweredSoftBevelBorder());
		adminButton.setIcon(adminImage);
		adminButton.setCursor(new Cursor(Cursor.HAND_CURSOR));
		adminButton.addActionListener(this);
		frame.add(adminButton);
		
	
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(800, 550);
		frame.setIconImage(framelogo.getImage());
		frame.setResizable(false);
		frame.setLayout (null);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	

	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==loginButton1)
		{
			loginAlert.setVisible(false);
			signUpButton1.setVisible(false);
			loginButton1.setVisible(false);
			loginButton2.setVisible(true);
			userName.setVisible(true);
			userPass.setVisible(true);
			tfUName.setVisible(true);
			pfUPass.setVisible(true);
			previous0.setVisible(true);
			tfUName.setText(null);
			pfUPass.setText(null);
		}
		
		else if(e.getSource()==signUpButton1)
		{
			
			new SignUp();
			frame.setVisible(false);
			
		}
		else if (e.getSource()==previous0)
		{
			loginAlert.setVisible(true);
			loginButton1.setVisible(true);
			loginButton2.setVisible(false);
			userName.setVisible(false);
			userPass.setVisible(false);
			tfUName.setVisible(false);
			pfUPass.setVisible(false);
			previous0.setVisible(false);
			signUpButton1.setVisible(true);
		}
		
		else if(e.getSource()==exitButton)
		{
			System.exit(0);
		}
		
		
		else if(e.getSource()==loginButton2)
		{  
			String user = tfUName.getText();
			String pass = pfUPass.getText();
            if (user.isEmpty()|| pass.isEmpty()) 
            {
                JOptionPane.showMessageDialog(null, "Please fill all of the fields.", "Warning!",
                        JOptionPane.WARNING_MESSAGE);
            }
            
            
            else {

                try {
                    String userNameS = "User Name : " + user;
                    String passwordS = "Password : " + pass;
                    BufferedReader reader = new BufferedReader(new FileReader("data\\.user_data.txt"));

                    int totalLines = 0;
                    while (reader.readLine() != null)
                        totalLines++;
                    reader.close();

                    for (int i = 0; i <= totalLines; i++) {
                        String line = Files.readAllLines(Paths.get("data\\.user_data.txt")).get(i);
                        if (line.equals(userNameS)) {
                            String line2 = Files.readAllLines(Paths.get("data\\.user_data.txt")).get((i + 1));
                            if (line2.equals(passwordS)) {
                                frame.setVisible(false);
                                new Home(user);
                                break;
                            }
                        }
                    }
                } 
                catch (Exception ex) {
                    JOptionPane.showMessageDialog(null, "Invalid User Name or Password!", "Warning!",
                            JOptionPane.WARNING_MESSAGE);
                }
            }
            
        }
		else if(e.getSource()==adminButton)
		{
			new AdminLogin();
			frame.setVisible(false);
		}
		
		
	}
	
}
