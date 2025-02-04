using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.StartPanel;

namespace Hungry_Heroes
{
    public partial class ManagerHomePage : Form, IManagerHomePage
    {
        private AdminHomeReuse adminHomeReuse;
        public ManagerHomePage()
        {
            InitializeComponent();
            adminHomeReuse = new AdminHomeReuse();
            pHome.Controls.Add(adminHomeReuse);
            pb1.Visible = true;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void bhome_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Home";
            //for button underline panel
            pb1.Visible = true;
            pb2.Visible = false;
            pb3.Visible = false;

            pb5.Visible = false;
            pb7.Visible = false;
            pb8.Visible = false;

            //for buttons & lebels
            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            tbsearch.Visible = false;

            //usercontrol add
            adminHomeReuse = new AdminHomeReuse();
            pHome.Controls.Add(adminHomeReuse);
        }

        public void bresturants_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Resturant";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = true;
            pb3.Visible = false;
            pb5.Visible = false;
            pb7.Visible = false;
            pb8.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = true;
            bdelete.Visible = true;
            bupdate.Visible = true;
            tbsearch.Visible = true;
            tbsearch.Enabled = true;

            //for Text
            badd.Text = "Add New Resturant";
            bdelete.Text = "Delete Resturant";
            bupdate.Text = "Update Resturant";
            tbsearch.PlaceholderText = "Delete or Upadate by Resturant_Owner Username";
            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for display text
            showResturnatDetails();
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
        }

        public void bprofile_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Users";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = true;
            pb5.Visible = false;
            pb7.Visible = false;
            pb8.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = true;
            bdelete.Visible = true;
            bupdate.Visible = true;
            tbsearch.Visible = true;
            tbsearch.Enabled = true;

            //for Text
            badd.Text = "Add New User";
            bdelete.Text = "Delete User";
            bupdate.Text = "Update User";
            tbsearch.PlaceholderText = "Delete or Upadate by Customer Username";
            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for display text
            showCustomertDetails();
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
        }
        public void bguest_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Guest Users";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb5.Visible = false;
            pb7.Visible = true;
            pb8.Visible = false;


            //for buttons & lebels
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            tbsearch.Enabled = false;
            tbsearch.PlaceholderText = "Disabled";
            tbsearch.Visible = true;
            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for display text
            showGuestUser();
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
        }

        public void blogdetails_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Login Details";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb5.Visible = false;
            pb7.Visible = false;
            pb8.Visible = true;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = true;
            bdelete.Visible = true;
            bupdate.Visible = true;
            tbsearch.Enabled = false;
            tbsearch.Visible = true;
            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for Text
            badd.Text = "Customer Only";
            bdelete.Text = "Resturant Only";
            bupdate.Text = "Guest Only";
            tbsearch.PlaceholderText = "Disabled";

            //for display text
            showLoginDetails();
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
        }
        public void blogout_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Logout";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb5.Visible = true;
            pb7.Visible = false;
            pb8.Visible = false;

            //for buttons & lebels
            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            tbsearch.Visible = false;
            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for confirming
            DialogResult result = MessageBox.Show("Are you sure to logout?", "Confirmation",
                               MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                this.Visible = false;
                ManagerLogin x = new ManagerLogin();
                x.Visible = true;
            }
            else
            {
                pb5.Visible = false;
                lTitles.Text = "Logout Cancelled!";
            }
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void badd_Click(object sender, EventArgs e)
        {
            //for login details of customer condition
            if (badd.Text == "Customer Only")
            {
                showCustomerLoginDetails();
            }
            else if (badd.Text == "Add New User")
            {
                CustomerAdd x = new CustomerAdd(this);
                x.Visible = true;
            }
            else if (badd.Text == "Add New Resturant")
            {
                ResturantOwnerAdd x = new ResturantOwnerAdd(this);
                x.Visible = true;
            }

        }

        public void bdelete_Click(object sender, EventArgs e)
        {

            //for login details of Resturant condition
            if (bdelete.Text == "Resturant Only")
            {
                showResturantOwnerLoginDetails();
            }
            else if (bdelete.Text == "Delete Resturant")
            {
                if (tbsearch.Text != string.Empty)
                {
                    //delete by username
                    string username = tbsearch.Text;
                    //Database Start
                    SqlConnection con = databaseConnection();
                    //checking if there is any user exist or not
                    SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@username", username);

                    SqlDataAdapter da = new SqlDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);
                    if (dt.Rows.Count > 0)
                    {
                        DialogResult result = MessageBox.Show("Are you sure you want to Delete this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showResturnatDetails();//refreshing page

                            tbsearch.Text = string.Empty;
                            tbsearch.Focus();
                        }

                    }
                    else
                    {
                        MessageBox.Show("User doesn't Exist!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }

                }
                else
                {
                    if (display.SelectedRows.Count > 0)
                    {
                        //implement code of delete by selecting
                        DataGridViewRow row = display.SelectedRows[0];
                        string username = row.Cells["USERNAME"].Value.ToString();
                        //Database Start
                        SqlConnection con = databaseConnection();

                        //conforming
                        DialogResult result = MessageBox.Show("Are you sure you want to Delete this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            SqlCommand cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showResturnatDetails();//refreshing page
                        }

                    }
                    else
                    {
                        MessageBox.Show("Write Username or Select an User first to delete!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }

                }
            }
            //another condition for button
            else if (bdelete.Text == "Delete User")
            {
                if (tbsearch.Text != string.Empty)
                {
                    //delete by username
                    string username = tbsearch.Text;
                    //Database Start
                    SqlConnection con = databaseConnection();
                    //checking if there is any user exist or not
                    SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@username", username);

                    SqlDataAdapter da = new SqlDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);
                    if (dt.Rows.Count > 0)
                    {
                        DialogResult result = MessageBox.Show("Are you sure you want to Delete this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            cmd = new SqlCommand("UPDATE CUSTOMER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username COLLATE Latin1_General_BIN", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showCustomertDetails();//refreshing page

                            tbsearch.Text = string.Empty;
                            tbsearch.Focus();
                        }

                    }
                    else
                    {
                        MessageBox.Show("User doesn't Exist!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }

                }
                else
                {
                    if (display.SelectedRows.Count > 0)
                    {
                        //implement code of delete by selecting
                        DataGridViewRow row = display.SelectedRows[0];
                        string username = row.Cells["USERNAME"].Value.ToString();
                        //Database Start
                        SqlConnection con = databaseConnection();

                        //conforming
                        DialogResult result = MessageBox.Show("Are you sure you want to Delete this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            SqlCommand cmd = new SqlCommand("UPDATE CUSTOMER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showCustomertDetails();//refreshing page
                        }

                    }
                    else
                    {
                        MessageBox.Show("Write Username or Select an User first to delete!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }

                }
            }

        }

        public void bupdate_Click(object sender, EventArgs e)
        {

            //for login details of Resturant condition
            if (bupdate.Text == "Guest Only")
            {
                showGuestLoginDetails();
            }
            else if (bupdate.Text == "Update User")
            {
                if (tbsearch.Text != string.Empty)
                {
                    string username = tbsearch.Text;
                    //database start
                    SqlConnection con = databaseConnection();
                    //checking if there is any user exist or not
                    SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@username", username);

                    SqlDataAdapter da = new SqlDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);
                    con.Close();
                    //database end

                    if (dt.Rows.Count > 0)
                    {
                        string name;
                        try { name = dt.Rows[0]["NAME"].ToString().Trim(); } catch { name = ""; }
                        string email;
                        try { email = dt.Rows[0]["EMAIL"].ToString().Trim(); } catch { email = ""; }
                        string password;
                        try { password = dt.Rows[0]["PASSWORD"].ToString().Trim(); } catch { password = ""; }
                        string birthday;
                        try { birthday = dt.Rows[0]["BIRTHDATE"].ToString(); } catch { birthday = ""; }
                        string address;
                        try { address = dt.Rows[0]["ADDRESS"].ToString().Trim(); } catch { address = ""; }
                        string sq;
                        try { sq = dt.Rows[0]["SQ"].ToString().Trim(); } catch { sq = ""; }
                        string sqa;
                        try { sqa = dt.Rows[0]["SQA"].ToString().Trim(); } catch { sqa = ""; }
                        string status;
                        try { status = dt.Rows[0]["ACCOUNT_STATUS"].ToString().Trim(); } catch { status = ""; }

                        CustomerAdd x = new CustomerAdd(this, name, username, email, address, birthday, sq, sqa, password, status);
                        x.Visible = true;

                    }
                    else
                    {
                        MessageBox.Show("User doesn't Exist!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
                else
                {
                    //update from selecting row
                    if (display.SelectedRows.Count > 0)
                    {
                        DataGridViewRow row = display.SelectedRows[0];
                        string username = row.Cells["USERNAME"].Value.ToString().Trim();
                        string name;
                        try { name = row.Cells["NAME"].Value.ToString().Trim(); } catch { name = ""; }
                        string email;
                        try { email = row.Cells["EMAIL"].Value.ToString().Trim(); } catch { email = ""; }
                        string password;
                        try { password = row.Cells["PASSWORD"].Value.ToString().Trim(); } catch { password = ""; }
                        string birthday;
                        try { birthday = row.Cells["BIRTHDATE"].Value.ToString(); } catch { birthday = ""; }
                        string address;
                        try { address = row.Cells["ADDRESS"].Value.ToString().Trim(); } catch { address = ""; }
                        string sq;
                        try { sq = row.Cells["SQ"].Value.ToString().Trim(); } catch { sq = ""; }
                        string sqa;
                        try { sqa = row.Cells["SQA"].Value.ToString().Trim(); } catch { sqa = ""; }
                        string status;
                        try { status = row.Cells["ACCOUNT_STATUS"].Value.ToString().Trim(); } catch { status = ""; }

                        CustomerAdd x = new CustomerAdd(this, name, username, email, address, birthday, sq, sqa, password, status);
                        x.Visible = true;
                    }
                    else
                    {
                        MessageBox.Show("Write Username or Select an User first to Update!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
            }
            else if (bupdate.Text == "Update Resturant")
            {
                if (tbsearch.Text != string.Empty)
                {
                    string username = tbsearch.Text;
                    //database start
                    SqlConnection con = databaseConnection();
                    //checking if there is any user exist or not
                    SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@username", username);

                    SqlDataAdapter da = new SqlDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);
                    con.Close();
                    //database end

                    if (dt.Rows.Count > 0)
                    {
                        string name;
                        try { name = dt.Rows[0]["NAME"].ToString().Trim(); } catch { name = ""; }
                        string email;
                        try { email = dt.Rows[0]["EMAIL"].ToString().Trim(); } catch { email = ""; }
                        string password;
                        try { password = dt.Rows[0]["PASSWORD"].ToString().Trim(); } catch { password = ""; }
                        string availability;
                        try { availability = dt.Rows[0]["AVAILABILITY"].ToString().Trim(); } catch { availability = ""; }
                        string phoneno;
                        try { phoneno = dt.Rows[0]["PHONE_NO"].ToString().Trim(); } catch { phoneno = ""; }
                        string resturantname;
                        try { resturantname = dt.Rows[0]["RESTURANT_NAME"].ToString().Trim(); } catch { resturantname = ""; }
                        string status;
                        try { status = dt.Rows[0]["ACCOUNT_STATUS"].ToString().Trim(); } catch { status = ""; }
                        string discount;
                        try { discount = dt.Rows[0]["DISCOUNT"].ToString().Trim(); } catch { discount = ""; }
                        ResturantOwnerAdd x = new ResturantOwnerAdd(this, name, username, email, phoneno, status, availability, password, resturantname, discount);
                        x.Visible = true;
                    }
                    else
                    {
                        MessageBox.Show("User doesn't Exist!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
                else
                {
                    //update from selecting row
                    if (display.SelectedRows.Count > 0)
                    {
                        DataGridViewRow row = display.SelectedRows[0];
                        string username = row.Cells["USERNAME"].Value.ToString().Trim();
                        string name;
                        try { name = row.Cells["NAME"].Value.ToString().Trim(); } catch { name = ""; }
                        string email;
                        try { email = row.Cells["EMAIL"].Value.ToString().Trim(); } catch { email = ""; }
                        string password;
                        try { password = row.Cells["PASSWORD"].Value.ToString().Trim(); } catch { password = ""; }
                        string availability;
                        try { availability = row.Cells["AVAILABILITY"].Value.ToString().Trim(); } catch { availability = ""; }
                        string phoneno;
                        try { phoneno = row.Cells["PHONE_NO"].Value.ToString().Trim(); } catch { phoneno = ""; }
                        string resturantname;
                        try { resturantname = row.Cells["RESTURANT_NAME"].Value.ToString().Trim(); } catch { resturantname = ""; }
                        string status;
                        try { status = row.Cells["ACCOUNT_STATUS"].Value.ToString().Trim(); } catch { status = ""; }
                        string discount;
                        try { discount = row.Cells["DISCOUNT"].Value.ToString().Trim(); } catch { discount = ""; }
                        ResturantOwnerAdd x = new ResturantOwnerAdd(this, name, username, email, phoneno, status, availability, password, resturantname, discount);
                        x.Visible = true;
                    }
                    else
                    {
                        MessageBox.Show("Write Username or Select an User first to Update!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
            }
            //rest of the condition


        }


        //all display methods
        public void showResturnatDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showCustomertDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showGuestUser()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM GUEST", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM LOGIN_DETAILS WHERE TYPE <> 'MANAGER'", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showCustomerLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM LOGIN_DETAILS WHERE TYPE = @type", con);
            cmd.Parameters.AddWithValue("@type", "CUSTOMER");
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showResturantOwnerLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM LOGIN_DETAILS WHERE TYPE = @type", con);
            cmd.Parameters.AddWithValue("@type", "RESTURANT_OWNER");
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showGuestLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT LOGIN_DETAILS_ID, TIME, TYPE FROM LOGIN_DETAILS WHERE TYPE = @type", con);
            cmd.Parameters.AddWithValue("@type", "GUEST");
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
    }
}
