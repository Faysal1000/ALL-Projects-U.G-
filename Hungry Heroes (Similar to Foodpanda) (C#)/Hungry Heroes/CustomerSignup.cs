using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Net.Mail;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.ListView;

namespace Hungry_Heroes
{
    public partial class CustomerSignup : Form, ICustomerSignup
    {
        public CustomerSignup()
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
            //value picking
            string name = tbname.Text;
            string username = tbusername.Text;
            string email = tbemail.Text;
            DateTime birthdate = birthdaypicker.Value;
            int birthyear = birthdaypicker.Value.Year;
            string address = tbaddress.Text;
            string sq = cbsq.Text;
            string sqa = tbsqa.Text;
            string pass = tbpass.Text;
            string confirmpass = tbcpass.Text;

            //captcha converting
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
                    MessageBox.Show("Please Enter Numeric Value to Ans", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }

            }
            int num1 = Convert.ToInt32(lcp1.Text);
            int num2 = Convert.ToInt32(lcp2.Text);
            int check = num1 + num2;

            if (name == "" || username == "" || email == "" || address == "" || sq == "" || sqa == "" || pass == "" || confirmpass == "")
            {
                MessageBox.Show("Please fill up all the fields", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else if (DateTime.Now.Year - birthyear < 12)
            {
                MessageBox.Show("You are too young to use this app", "Warning",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
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
                //check wheather username already exist or not
                //Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
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
                    //check email already exist or not
                    cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE EMAIL =@email", con);
                    cmd.Parameters.AddWithValue("@email", email);

                    SqlDataAdapter da2 = new SqlDataAdapter(cmd);
                    DataTable dt2 = new DataTable();
                    da2.Fill(dt2);
                    //...
                    if (dt2.Rows.Count > 0)
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
                                string query = "INSERT INTO CUSTOMER VALUES (@username, @name, @email, @password, @birthdate, @address, @sq, @sqa, @status)";
                                cmd = new SqlCommand(query, con);
                                cmd.Parameters.AddWithValue("@username", username);
                                cmd.Parameters.AddWithValue("@name", name);
                                cmd.Parameters.AddWithValue("@email", email);
                                cmd.Parameters.AddWithValue("@password", pass);
                                cmd.Parameters.AddWithValue("@birthdate", birthdate);
                                cmd.Parameters.AddWithValue("@address", address);
                                cmd.Parameters.AddWithValue("@sq", sq);
                                cmd.Parameters.AddWithValue("@sqa", sqa);
                                cmd.Parameters.AddWithValue("@status", "Active");
                                cmd.ExecuteNonQuery();
                                con.Close();
                                //database end

                                MessageBox.Show("Signup Complete!", "Success",
                                                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                                //clear everything
                                tbname.Text = string.Empty;
                                tbusername.Text = string.Empty;
                                tbemail.Text = string.Empty;
                                birthdaypicker.Value = DateTime.Now;
                                tbaddress.Text = string.Empty;
                                cbsq.SelectedIndex = -1;
                                tbsqa.Text = string.Empty;
                                tbcans.Text = string.Empty;
                                tbpass.Text = string.Empty;
                                tbcpass.Text = string.Empty;

                                //generating random number for captcha
                                Random random = new Random();
                                num1 = random.Next(10, 21);
                                num2 = random.Next(10, 21);
                                lcp1.Text = num1.ToString();
                                lcp2.Text = num2.ToString();
                            }
                            else
                            {
                                MessageBox.Show("Password Should Contain at least 1 Upper letter and 1 Digit!", "Alert",
                                               MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                            }
                        }
                        else
                        {
                            MessageBox.Show("Your password should contain atleast 6 digit!", "Alert",
                                                MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                        }
                    }

                }

            }
            else
            {
                MessageBox.Show("Wrong Inputs", "Invalid Input",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
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
