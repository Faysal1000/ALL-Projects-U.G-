using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace Hungry_Heroes
{
    public partial class CustomerEdit : Form, ICustomerEdit
    {
        private string name, username, email, address, birthday, sq, sqa, password;
        private CustomerHomePage customerHomePage;
        private ProfileReuse profileReuse;
        public CustomerEdit(ProfileReuse profileReuse,CustomerHomePage customerHomePage, string name, string username, string email, string address, string birthday, string sq, string sqa, string password)
        {
            InitializeComponent();
            this.profileReuse = profileReuse;
            this.customerHomePage = customerHomePage;
            this.name = name;
            this.username = username;
            this.email = email;
            this.address = address;
            this.birthday = birthday;
            this.sq = sq;
            this.sqa = sqa;
            this.password = password;
            customerHomePage.Enabled = false;
            customerHomePage.Opacity = 0.5;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void CustomerEdit_Load(object sender, EventArgs e)
        {
            tbname.Text = name;
            tbusername.Text = username;
            tbemail.Text = email;
            tbpass.Text = password;
            tbsqa.Text = sqa;
            tbaddress.Text = address;
            try { birthdaypicker.Value = DateTime.Parse(birthday); } catch { }

            //sq check
            if (sq == "What is your favourite color?")
            {
                cbsq.SelectedIndex = 0;
            }
            else if (sq == "What is your birth place?")
            {
                cbsq.SelectedIndex = 1;
            }
            else if (sq == "What is your hobby?")
            {
                cbsq.SelectedIndex = 2;
            }
            else if (sq == "What is your crush name?")
            {
                cbsq.SelectedIndex = 3;
            }
            else if (sq == "What is your best friend name?")
            {
                cbsq.SelectedIndex = 4;
            }
        }

        public void bupdate_Click(object sender, EventArgs e)
        {
            //Database Start
            SqlConnection con = databaseConnection();
            //check wheather username already exist or not
            SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
            cmd.Parameters.AddWithValue("@username", tbusername.Text);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            if (dt.Rows.Count > 0 && tbusername.Text != username)
            {
                MessageBox.Show("Username Already Exist! Choose another one.", "Error!",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {

                if (tbusername.Text == "" || tbname.Text == "" || tbemail.Text == "" || tbaddress.Text == "" || cbsq.Text == "" || tbsqa.Text == "" || tbpass.Text == "")
                {
                    MessageBox.Show("Please fill up all the fields", "Information",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else if (IsValidEmail(tbemail.Text) == false)
                {
                    MessageBox.Show("Email Address is not Valid!", "Warning!",
                        MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                else
                {
                    cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE EMAIL = @email COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@email", tbemail.Text);
                    da = new SqlDataAdapter(cmd);
                    DataTable dt2 = new DataTable();
                    da.Fill(dt2);
                    if (dt2.Rows.Count > 0 && tbemail.Text != email)
                    {
                        MessageBox.Show("Email Address Already Exist!", "Warning!",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                    else
                    {
                        try
                        {
                            cmd = new SqlCommand("UPDATE CUSTOMER SET NAME=@name,USERNAME=@newusername,EMAIL=@email," +
                            "BIRTHDATE=@birthdate,ADDRESS=@address,SQ=@sq, SQA=@sqa,PASSWORD=@password " +
                            "WHERE USERNAME =@username COLLATE Latin1_General_BIN", con);

                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@newusername", tbusername.Text);
                            cmd.Parameters.AddWithValue("@name", tbname.Text);
                            cmd.Parameters.AddWithValue("@email", tbemail.Text);
                            cmd.Parameters.AddWithValue("@birthdate", birthdaypicker.Value);
                            cmd.Parameters.AddWithValue("@address", tbaddress.Text);
                            cmd.Parameters.AddWithValue("@sq", cbsq.Text);
                            cmd.Parameters.AddWithValue("@sqa", tbsqa.Text);
                            cmd.Parameters.AddWithValue("@password", tbpass.Text);
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            MessageBox.Show("Data Updated Successfully", "Success!",
                                        MessageBoxButtons.OK, MessageBoxIcon.Information);

                            this.Visible = false;
                            customerHomePage.Opacity = 1;
                            customerHomePage.Enabled = true;
                            profileReuse.profileReuseRefresh();
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Give Correct And All the Data!"+ex.Message, "Error!",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }

                }

            }
        }
        public bool IsValidEmail(string emailAddress)
        {
            var pattern = @"^[a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$";

            var regex = new Regex(pattern);

            return regex.IsMatch(emailAddress);
        }

        public void bcancel_Click(object sender, EventArgs e)
        {
            customerHomePage.Opacity = 1;
            customerHomePage.Enabled = true;
            this.Visible=false;
        }
    }
}
