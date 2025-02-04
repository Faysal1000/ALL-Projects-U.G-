using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Hungry_Heroes
{
    public partial class ResturantSignup : Form, IResturantSignup
    {
        public ResturantSignup()
        {
            InitializeComponent();
            //generating random number for captcha
            Random random = new Random();
            int num1 = random.Next(10, 21);
            int num2 = random.Next(10, 21);
            lcp1.Text = num1.ToString();
            lcp2.Text = num2.ToString();
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }

        //For checking password is valid or not
        //Pass should contain 6 character one upper and one digit
        public bool IsPasswordValid(string password)
        {
            bool hasCapitalLetter = false;
            bool hasNumber = false;
            foreach (char c in password)
            {
                if (char.IsUpper(c))
                {
                    hasCapitalLetter = true;
                }
                else if (char.IsDigit(c))
                {
                    hasNumber = true;
                }

                if (hasCapitalLetter && hasNumber)
                {
                    return true; //Password is ok
                }
            }
            return false; //pass is not ok
        }

        public void bsignup_Click(object sender, EventArgs e)
        {
            string name = tbname.Text;
            string username = tbusername.Text;
            string email = tbemail.Text;
            string phoneno = tbphone.Text;
            string resturantname = tbresname.Text;
            string pass = tbpass.Text;
            string confirmpass = tbcpass.Text;

            int ans = 0;
            if (tbcans.Text != "")
            {
                try
                {
                    ans = int.Parse(tbcans.Text);
                }
                catch (Exception a)
                {
                    ans = -1;
                    MessageBox.Show("Please Enter Numeric Value to Ans",
                        "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }

            }
            int num1 = Convert.ToInt32(lcp1.Text);
            int num2 = Convert.ToInt32(lcp2.Text);
            int check = num1 + num2;
            if (ans != -1)
            {
                if (name == "" || username == "" || email == "" || phoneno == "" || resturantname == "" || pass == "" || confirmpass == "")
                {
                    MessageBox.Show("Please fill up all the fields", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else if (phoneno.Length != 11)
                {
                    MessageBox.Show("Your Phone no should have 11 Character", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else if (IsValidEmail(email) == false)
                {
                    MessageBox.Show("Email Address is not Valid!", "Warning!",
                        MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                else if (confirmpass == pass && check != ans)
                {
                    if (ans != -1)
                    {
                        MessageBox.Show("Capta Ans. is wrong", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
                else if (confirmpass != pass && check == ans)
                {
                    MessageBox.Show("Password didn't match", "Invalid Input",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else if (confirmpass == pass && check == ans)
                {
                    //checking phone no wheather it is number or other character
                    long phonecheck = -1;
                    if (phoneno.Length == 11)
                    {
                        try
                        {
                            phonecheck = long.Parse(phoneno);
                        }
                        catch (Exception a)
                        {
                            MessageBox.Show("Give Numerical value to phone numbers", "Information",
                                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                            phonecheck = -1;
                        }
                    }
                    if (phonecheck != -1)
                    {
                        //check wheather username already exist or not
                        //Database start
                        SqlConnection con = databaseConnection();
                        SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                        cmd.Parameters.AddWithValue("@username", username);

                        SqlDataAdapter da = new SqlDataAdapter(cmd);
                        DataTable dt = new DataTable();
                        da.Fill(dt);
                        //....
                        if (dt.Rows.Count > 0)
                        {
                            MessageBox.Show("Username Not Available! Choose another one!", "Alert",
                                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                            tbusername.Text = string.Empty;
                            tbusername.Focus();
                        }
                        else
                        {
                            //check resturant name is unique or not
                            cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                            cmd.Parameters.AddWithValue("@username", username);

                            da = new SqlDataAdapter(cmd);
                            DataTable dt2 = new DataTable();
                            da.Fill(dt2);
                            //....

                            if (dt2.Rows.Count > 0)
                            {
                                MessageBox.Show("Resturant name Not Available! Choose another one!", "Alert",
                                                MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                                tbresname.Text = string.Empty;
                                tbresname.Focus();
                            }
                            else
                            {
                                //check email  is unique or not
                                cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE EMAIL=@email", con);
                                cmd.Parameters.AddWithValue("@email", email);

                                da = new SqlDataAdapter(cmd);
                                DataTable dt3 = new DataTable();
                                da.Fill(dt3);
                                //....

                                if (dt3.Rows.Count > 0)
                                {
                                    MessageBox.Show("Email Adress already Exist! Choose another one!", "Alert",
                                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                                    tbemail.Text = string.Empty;
                                    tbemail.Focus();
                                }
                                else
                                {
                                    if (pass.Length >= 6)
                                    {
                                        if (IsPasswordValid(pass))
                                        {
                                            // Password meets the criteria
                                            //let user sign in now
                                            //add details in data base
                                            string query = "INSERT INTO RESTURANT_OWNER VALUES (@username, @name, @password, @email, @availability, @phoneno, @resturantname, @totalorder, @totalrating, @status, @discount)";
                                            cmd = new SqlCommand(query, con);
                                            cmd.Parameters.AddWithValue("@username", username);
                                            cmd.Parameters.AddWithValue("@name", name);
                                            cmd.Parameters.AddWithValue("@password", pass);
                                            cmd.Parameters.AddWithValue("@email", email);
                                            cmd.Parameters.AddWithValue("@availability", "Yes");
                                            cmd.Parameters.AddWithValue("@phoneno", phoneno);
                                            cmd.Parameters.AddWithValue("@resturantname", resturantname);
                                            cmd.Parameters.AddWithValue("@totalorder", 0);
                                            cmd.Parameters.AddWithValue("@totalrating", 0);
                                            cmd.Parameters.AddWithValue("@status", "Active");
                                            cmd.Parameters.AddWithValue("@discount", 0);
                                            cmd.ExecuteNonQuery();
                                            con.Close();
                                            //Database end

                                            MessageBox.Show("Signup Complete!", "Success",
                                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                                            //clear everything
                                            tbname.Text = string.Empty;
                                            tbusername.Text = string.Empty;
                                            tbemail.Text = string.Empty;
                                            tbphone.Text = string.Empty;
                                            tbresname.Text = string.Empty;
                                            tbpass.Text = string.Empty;
                                            tbcpass.Text = string.Empty;
                                            tbcans.Text = string.Empty;

                                            //generating random number for captcha
                                            Random random = new Random();
                                            num1 = random.Next(10, 21);
                                            num2 = random.Next(10, 21);
                                            lcp1.Text = num1.ToString();
                                            lcp2.Text = num2.ToString();
                                        }
                                        else
                                        {
                                            //Password didn't have 1 upper case and 1 digit 
                                            MessageBox.Show("Password Should Contain at least 1 Upper letter and 1 Digit!", "Alert",
                                                           MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                                        }
                                    }
                                    else
                                    {
                                        //password is not 6 digit long
                                        MessageBox.Show("Your password should contain atleast 6 digit!", "Alert",
                                                            MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                                    }
                                }
                                
                            }
                        }

                    }

                }
            }
        }

        public void bprevious_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            First_page x = new First_page();
            x.Visible = true;
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void brecapta_Click(object sender, EventArgs e)
        {
            Random random = new Random();
            int num1 = random.Next(10, 21);
            int num2 = random.Next(10, 21);
            lcp1.Text = num1.ToString();
            lcp2.Text = num2.ToString();
        }
        public bool IsValidEmail(string emailAddress)
        {
            var pattern = @"^[a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$";

            var regex = new Regex(pattern);

            return regex.IsMatch(emailAddress);
        }
    }
}
