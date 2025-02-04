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

namespace Hungry_Heroes
{
    public partial class AdminHomepage : Form, IAdminHomePage
    {
        private DataTable DT;
        private AdminHomeReuse adminHomeReuse;
        public AdminHomepage()
        {
            InitializeComponent();
            DT = new DataTable();
            adminHomeReuse = new AdminHomeReuse();
            pHome.Controls.Add(adminHomeReuse);
            adminHomeReuse.Location = new Point(145, 87);
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
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            blogmanager.Visible = false;
            tbsearch.Visible = false;

            //usercontrol add
            pHome.Controls.Remove(adminHomeReuse);
            pHome.Controls.Add(adminHomeReuse);
            // Set the user control's Dock property to fill the parent panel
            adminHomeReuse.Location = new Point(145, 87);

        }

        public void bresturants_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Resturant";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = true;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = true;
            bupdate.Visible = true;
            blogmanager.Visible = false;
            tbsearch.Visible = true;
            tbsearch.Enabled = true;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);
            //for Text
            bdelete.Text = "Deactivate Resturant";
            bupdate.Text = "Update Resturant";
            tbsearch.PlaceholderText = "Delete or Upadate by Resturant_Owner Username";

            //for display text
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
            showResturnatDetails();
        }

        public void bprofile_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Users";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = true;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = true;
            bupdate.Visible = true;
            blogmanager.Visible = false;
            tbsearch.Visible = true;
            tbsearch.Enabled = true;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);

            //for Text
            bdelete.Text = "Deactivate User";
            bupdate.Text = "Update User";
            tbsearch.PlaceholderText = "Delete or Upadate by Customer Username";

            //for display text
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
            showCustomertDetails();

        }
        public void bmanager_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Managers";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = true;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = true;
            bupdate.Visible = true;
            blogmanager.Visible = false;
            tbsearch.Visible = true;
            tbsearch.Enabled = true;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);

            //for Text
            bdelete.Text = "Deactivate Manager";
            bupdate.Text = "Update Manager";
            tbsearch.PlaceholderText = "Delete or Upadate by Manager Username";

            //for display text
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            showManagerDetails();

        }

        public void bguest_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Guest Users";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = true;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            blogmanager.Visible = false;
            tbsearch.Enabled = false;
            tbsearch.PlaceholderText = "Disabled";
            tbsearch.Visible = true;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);

            //for display text
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            showGuestUser();


        }
        public void bprofits_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Login Details";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = true;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = true;
            badd.Visible = true;
            bdelete.Visible = true;
            bupdate.Visible = true;
            blogmanager.Visible = true;
            tbsearch.Enabled = false;
            tbsearch.Visible = true;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);

            //button text
            badd.Text = "Customer Only";
            bdelete.Text = "Resturant Only";
            bupdate.Text = "Guest Only";
            tbsearch.PlaceholderText = "Disabled";

            //for display text
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            showLoginDetails();

        }
        public void blogout_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Logout";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = true;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            bupdate.Visible = false;
            blogmanager.Visible = false;
            tbsearch.Visible = false;

            //usercontrol remove
            pHome.Controls.Remove(adminHomeReuse);

            //for confirming
            DialogResult result = MessageBox.Show("Are you sure to logout?", "Confirmation",
                               MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                this.Visible = false;
                AdminLogin x = new AdminLogin();
                x.Visible = true;
            }
            else
            {
                pb5.Visible = false;
                lTitles.Text = "Logout Canceled!";
            }
        }


        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void badd_Click(object sender, EventArgs e)
        {
            showCustomerLoginDetails();
        }

        public void bdelete_Click(object sender, EventArgs e)
        {

            //for login details of Resturant condition
            if (bdelete.Text == "Resturant Only")
            {
                showResturantOwnerLoginDetails();
            }
            else if (bdelete.Text == "Deactivate Resturant")
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
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
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
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
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
                        MessageBox.Show("Write Username or Select an User first to Deactivate!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }

                }
            }
            //another condition for button
            else if (bdelete.Text == "Deactivate User")
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
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            cmd = new SqlCommand("UPDATE CUSTOMER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
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
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
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
                        MessageBox.Show("Write Username or Select an User first to Deactivate!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }

                }
            }
            else if (bdelete.Text == "Deactivate Manager")
            {
                //for manager
                if (tbsearch.Text != string.Empty)
                {
                    //delete by username
                    string username = tbsearch.Text;
                    //Database Start
                    SqlConnection con = databaseConnection();
                    //checking if there is any user exist or not
                    SqlCommand cmd = new SqlCommand("SELECT * FROM MANAGER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@username", username);

                    SqlDataAdapter da = new SqlDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);
                    if (dt.Rows.Count > 0)
                    {
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            cmd = new SqlCommand("UPDATE MANAGER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showManagerDetails();//refreshing page

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
                        DialogResult result = MessageBox.Show("Are you sure you want to Deactivate this User?", "Question",
                                                        MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                        if (result == DialogResult.Yes)
                        {
                            SqlCommand cmd = new SqlCommand("UPDATE MANAGER SET ACCOUNT_STATUS=@status WHERE USERNAME =@username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@status", "Deactive");
                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database end

                            showManagerDetails();//refreshing page
                        }

                    }
                    else
                    {
                        MessageBox.Show("Write Username or Select an User first to Deactivate!", "Warning",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }

                }
            }

        }
        public void bupdate_Click_1(object sender, EventArgs e)
        {
            //for login details of Resturant condition
            if (bupdate.Text == "Guest Only")
            {
                showGuestLoginDetails();
            }
            else if (bupdate.Text == "Update Resturant")
            {
                //resturant updating
                // Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER", con);
                SqlDataAdapter da = new SqlDataAdapter(cmd);

                SqlCommandBuilder commandBuilder = new SqlCommandBuilder(da);
                try
                {
                    da.Update(DT);
                    
                    showResturnatDetails();
                    MessageBox.Show("Data updated successfully!", "Successfull",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error updating data: " + ex.Message, "Error!",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                con.Close();
                //database end
            }
            else if (bupdate.Text == "Update User")
            {
                //CUSTOMER updating
                // Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER", con);
                SqlDataAdapter da = new SqlDataAdapter(cmd);

                SqlCommandBuilder commandBuilder = new SqlCommandBuilder(da);

                try
                {
                    da.Update(DT);
                    showCustomertDetails();
                    MessageBox.Show("Data updated successfully!", "Successfull",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error updating data: " + ex.Message, "Error!",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                con.Close();
                //database end
            }
            else if (bupdate.Text == "Update Manager")
            {
                //MANAGER updating
                // Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM MANAGER", con);
                SqlDataAdapter da = new SqlDataAdapter(cmd);

                SqlCommandBuilder commandBuilder = new SqlCommandBuilder(da);

                try
                {
                    da.Update(DT);
                    showManagerDetails();
                    MessageBox.Show("Data updated successfully!", "Successfull",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error updating data: " + ex.Message, "Error!",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                con.Close();
                //database end
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM LOGIN_DETAILS", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
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
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showManagerLoginDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM LOGIN_DETAILS WHERE TYPE = @type", con);
            cmd.Parameters.AddWithValue("@type", "MANAGER");
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void showManagerDetails()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM MANAGER", con);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DT = new DataTable();
            da.Fill(DT);
            display.DataSource = DT;
            display.ClearSelection();
            con.Close();
            //database end here
        }

        public void blogmanager_Click(object sender, EventArgs e)
        {
            showManagerLoginDetails();
        }
    }
}
