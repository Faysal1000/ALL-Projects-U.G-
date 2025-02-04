using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;

namespace Hungry_Heroes
{
    public partial class CustomerAdd : Form, ICustomerAdd
    {
        private string name, username, email, address, birthday, sq, sqa, password, status;
        private ManagerHomePage managerHomePage;
        private string flag = "Update";
        public CustomerAdd(ManagerHomePage managerHomePage, string name, string username, string email, string address, string birthday, string sq, string sqa, string password, string status)
        {
            this.name = name;
            this.username = username;
            this.email = email;
            this.address = address;
            this.birthday = birthday;
            this.sq = sq;
            this.sqa = sqa;
            this.password = password;
            this.status = status;
            this.managerHomePage = managerHomePage;
            managerHomePage.Opacity = 0.5;
            managerHomePage.Enabled = false;
            InitializeComponent();
            bupdate.Visible = true;
        }
        public CustomerAdd(ManagerHomePage managerHomePage)
        {
            this.managerHomePage = managerHomePage;
            this.flag = "Add";
            managerHomePage.Opacity = 0.5;
            managerHomePage.Enabled = false;
            InitializeComponent();
            badd.Visible = true;
            ltitle.Text = "Add new User (Customer)";
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
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
            else if (IsValidEmail(tbemail.Text) == false)
            {
                MessageBox.Show("Email Address is not Valid!", "Warning!",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else
            {

                if (tbusername.Text == "" || tbname.Text == "" || tbemail.Text == "" || tbaddress.Text == "" || cbsq.Text == "" || tbsqa.Text == "" || tbpass.Text == "" || cbstatus.Text == "")
                {
                    MessageBox.Show("Please fill up all the fields", "Information",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
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
                            "BIRTHDATE=@birthdate,ADDRESS=@address,SQ=@sq, SQA=@sqa,PASSWORD=@password, ACCOUNT_STATUS=@status " +
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
                            cmd.Parameters.AddWithValue("@status", cbstatus.Text);
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            MessageBox.Show("Data Updated Successfully", "Success!",
                                        MessageBoxButtons.OK, MessageBoxIcon.Information);

                            managerHomePage.Enabled = true;
                            managerHomePage.showCustomertDetails();
                            managerHomePage.Opacity = 1;
                            managerHomePage.tbsearch.Text=string.Empty;
                            this.Visible = false;
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Give Correct And All the Data!", "Error!",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                        
                }

            }
        }

        public void updateDataLoad()
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
            //status check
            if (status == "Active")
            {
                cbstatus.SelectedIndex = 0;
            }
            else
            {
                cbstatus.SelectedIndex = 1;
            }
        }
        public void CustomerAdd_Load(object sender, EventArgs e)
        {
            if (flag == "Update")
            {
                updateDataLoad();
            }

        }

        public void bcancel_Click(object sender, EventArgs e)
        {
            managerHomePage.Enabled = true;
            managerHomePage.Opacity = 1;
            managerHomePage.tbsearch.Text = string.Empty;
            this.Visible = false;
        }

        public void badd_Click(object sender, EventArgs e)
        {
            //Database Start
            SqlConnection con = databaseConnection();
            //check wheather username already exist or not
            SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
            cmd.Parameters.AddWithValue("@username", tbusername.Text);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            if (dt.Rows.Count > 0)
            {
                MessageBox.Show("Username Already Exist! Choose another one.", "Error!",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                string name = tbname.Text;
                string username = tbusername.Text;
                string email = tbemail.Text;
                DateTime birthdate = birthdaypicker.Value;
                int birthyear = birthdaypicker.Value.Year;
                string address = tbaddress.Text;
                string sq = cbsq.Text;
                string sqa = tbsqa.Text;
                string pass = tbpass.Text;
                string status = cbstatus.Text;
                if (name == "" || username == "" || email == "" || address == "" || sq == "" || sqa == "" || pass == "" || status == "")
                {
                    MessageBox.Show("Please fill up all the fields", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
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
                    if (dt2.Rows.Count > 0 )
                    {
                        MessageBox.Show("Email Address Already Exist!", "Warning!",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                    else
                    {
                        try
                        {
                            //inserting Value
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
                            cmd.Parameters.AddWithValue("@status", status);
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            MessageBox.Show("New User Added", "Information",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                            managerHomePage.showCustomertDetails();
                            managerHomePage.Enabled = true;
                            managerHomePage.Opacity = 1;
                            this.Visible = false;
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Give Correct And All the Data!", "Error!",
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
    }
}
