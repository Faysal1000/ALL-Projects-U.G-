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
    public partial class ResturantOwnerAdd : Form, IResturantOwnerAdd
    {
        string name, username, email, contactno, status, availability, password, resturantname, discount;
        ManagerHomePage managerHomePage;
        string flag = "Update";
        public ResturantOwnerAdd(ManagerHomePage managerHomePage, string name, string username, string email, string contacto, string status, string availability, string password, string resturantname, string discount)
        {
            this.managerHomePage = managerHomePage;
            this.name = name;
            this.username = username;
            this.email = email;
            this.contactno = contacto;
            this.status = status;
            this.availability = availability;
            this.password = password;
            this.resturantname = resturantname;
            this.discount = discount;
            InitializeComponent();
            bupdate.Visible = true;
            managerHomePage.Enabled = false;
            managerHomePage.Opacity = 0.5;
        }
        public ResturantOwnerAdd(ManagerHomePage managerHomePage)
        {
            InitializeComponent();
            this.managerHomePage = managerHomePage;
            badd.Visible = true;
            managerHomePage.Enabled = false;
            managerHomePage.Opacity = 0.5;
            flag = "Add";
            ltitle.Text = "Add New Resturant Owner";
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void updateDataLoad()
        {
            tbname.Text = name;
            tbusername.Text = username;
            tbemail.Text = email;
            tbphone.Text = contactno;
            if (string.Equals(status, "Active", StringComparison.OrdinalIgnoreCase))
            {
                rbactive.Checked = true;
            }
            else
            {
                rbdeactive.Checked = true;
            }
            if (string.Equals(availability, "Yes", StringComparison.OrdinalIgnoreCase))
            {
                yes.Checked = true;
            }
            else
            {
                no.Checked = true;
            }
            tbpass.Text = password;
            tbresname.Text = resturantname;
            tbdiscount.Text = discount;
        }
        public void ResturantOwnerAdd_Load(object sender, EventArgs e)
        {
            if (flag == "Update")
            {
                updateDataLoad();
            }
        }

        public void badd_Click(object sender, EventArgs e)
        {
            //Database Start
            SqlConnection con = databaseConnection();
            //check wheather username already exist or not
            SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
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
                cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname COLLATE Latin1_General_BIN", con);
                cmd.Parameters.AddWithValue("@resturantname", tbresname.Text);
                da = new SqlDataAdapter(cmd);
                DataTable dt2 = new DataTable();
                da.Fill(dt2);
                if (dt2.Rows.Count > 0)
                {
                    MessageBox.Show("Resturant Name Already Exist! Choose another one.", "Error!",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                else
                {
                    if (tbusername.Text == "" || tbname.Text == "" || tbemail.Text == "" || tbphone.Text == "" || tbresname.Text == "" || tbpass.Text == "" || tbdiscount.Text == "")
                    {
                        MessageBox.Show("Please fill up all the fields", "Information",
                                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        if (IsValidEmail(tbemail.Text) == false)
                        {
                            MessageBox.Show("Email Address is not Valid!", "Warning!",
                                MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        }
                        else
                        {
                            cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE EMAIL = @email COLLATE Latin1_General_BIN", con);
                            cmd.Parameters.AddWithValue("@email", tbemail.Text);
                            da = new SqlDataAdapter(cmd);
                            DataTable dt3 = new DataTable();
                            da.Fill(dt3);

                            if (dt3.Rows.Count > 0)
                            {
                                MessageBox.Show("Email Address Already Exist!", "Warning!",
                                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                            }
                            else
                            {
                                try
                                {
                                    //inserting Value
                                    string query = "INSERT INTO RESTURANT_OWNER VALUES (@username, @name, @password, @email , " +
                                        "@availability, @phoneno, @resturantname, @totalorder, @totalrating,@status,@discount)";
                                    cmd = new SqlCommand(query, con);
                                    cmd.Parameters.AddWithValue("@username", tbusername.Text);
                                    cmd.Parameters.AddWithValue("@name", tbname.Text);
                                    cmd.Parameters.AddWithValue("@email", tbemail.Text);
                                    cmd.Parameters.AddWithValue("@password", tbpass.Text);
                                    //availability
                                    if (yes.Checked)
                                    {
                                        cmd.Parameters.AddWithValue("@availability", "Yes");
                                    }
                                    else
                                    {
                                        cmd.Parameters.AddWithValue("@availability", "No");
                                    }
                                    cmd.Parameters.AddWithValue("@phoneno", tbphone.Text);
                                    cmd.Parameters.AddWithValue("@resturantname", tbresname.Text);
                                    cmd.Parameters.AddWithValue("@totalorder", 0);
                                    cmd.Parameters.AddWithValue("@totalrating", 0);
                                    //status
                                    if (rbactive.Checked)
                                    {
                                        cmd.Parameters.AddWithValue("@status", "Active");
                                    }
                                    else
                                    {
                                        cmd.Parameters.AddWithValue("@status", "Deactive");
                                    }
                                    //discount check
                                    if (int.TryParse(tbdiscount.Text, out int intDiscount))
                                    {
                                        cmd.Parameters.AddWithValue("@discount", intDiscount);
                                    }
                                    else
                                    {
                                        cmd.Parameters.AddWithValue("@discount", 0);
                                        MessageBox.Show("Error in Data! Default value 0 added in Discount!", "Infromation",
                                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                                    }

                                    cmd.ExecuteNonQuery();
                                    con.Close();
                                    //database end

                                    MessageBox.Show("New User Added!", "Information",
                                        MessageBoxButtons.OK, MessageBoxIcon.Information);

                                    managerHomePage.showResturnatDetails();
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

            }
        }

        public void bupdate_Click(object sender, EventArgs e)
        {
            //Database Start
            SqlConnection con = databaseConnection();
            //check wheather username already exist or not
            SqlCommand cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
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

                if (tbusername.Text == "" || tbname.Text == "" || tbemail.Text == "" || tbphone.Text == "" || tbresname.Text == "" || tbpass.Text == "" || tbdiscount.Text == "")
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
                    cmd = new SqlCommand("SELECT * FROM RESTURANT_OWNER WHERE EMAIL = @email COLLATE Latin1_General_BIN", con);
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
                            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET NAME=@name,USERNAME=@newusername,EMAIL=@email,PASSWORD=@password," +
                                "AVAILABILITY=@availability,PHONE_NO=@phoneno, RESTURANT_NAME=@resturantname, ACCOUNT_STATUS=@status,DISCOUNT=@discount " +
                                "WHERE USERNAME =@username COLLATE Latin1_General_BIN", con);

                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@newusername", tbusername.Text);
                            cmd.Parameters.AddWithValue("@name", tbname.Text);
                            cmd.Parameters.AddWithValue("@email", tbemail.Text);
                            //availability
                            if (yes.Checked)
                            {
                                cmd.Parameters.AddWithValue("@availability", "Yes");
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@availability", "No");
                            }
                            //status
                            if (rbactive.Checked)
                            {
                                cmd.Parameters.AddWithValue("@status", "Active");
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@status", "Deactive");
                            }
                            cmd.Parameters.AddWithValue("@phoneno", tbphone.Text);
                            cmd.Parameters.AddWithValue("@resturantname", tbresname.Text);
                            cmd.Parameters.AddWithValue("@password", tbpass.Text);
                            //discount check
                            if (int.TryParse(tbdiscount.Text, out int intDiscount))
                            {
                                cmd.Parameters.AddWithValue("@discount", intDiscount);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@discount", 0);
                                MessageBox.Show("Error in Discount Data! Default value 0 added in Discount!", "Infromation",
                                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                            }
                            cmd.ExecuteNonQuery();
                            //extra works if user updated the resturant name
                            if (resturantname != tbresname.Text)
                            {
                                cmd = new SqlCommand("UPDATE BILL SET RESTURANT_NAME= @newresname WHERE RESTURANT_NAME = @resturantname COLLATE Latin1_General_BIN", con);
                                cmd.Parameters.AddWithValue("@newresname", tbresname.Text);
                                cmd.Parameters.AddWithValue("@resturantname", resturantname);
                                cmd.ExecuteNonQuery();
                                cmd = new SqlCommand("UPDATE MENU SET RESTURANT_NAME= @newresname WHERE RESTURANT_NAME = @resturantname COLLATE Latin1_General_BIN", con);
                                cmd.Parameters.AddWithValue("@newresname", tbresname.Text);
                                cmd.Parameters.AddWithValue("@resturantname", resturantname);
                                cmd.ExecuteNonQuery();
                            }
                            con.Close();
                            //database end
                            //
                            MessageBox.Show("Data Updated Successfully", "Success!",
                                        MessageBoxButtons.OK, MessageBoxIcon.Information);

                            managerHomePage.Enabled = true;
                            managerHomePage.showResturnatDetails();
                            managerHomePage.tbsearch.Text = string.Empty;
                            managerHomePage.Opacity = 1;
                            this.Visible = false;
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Give Correct And All the Data! "+ex.Message, "Error!",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                    }
                }
            }
        }

        public void bcancel_Click(object sender, EventArgs e)
        {
            managerHomePage.Enabled = true;
            managerHomePage.Opacity = 1;
            managerHomePage.tbsearch.Text = string.Empty;
            this.Visible = false;
        }
        public bool IsValidEmail(string emailAddress)
        {
            var pattern = @"^[a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$";

            var regex = new Regex(pattern);

            return regex.IsMatch(emailAddress);
        }
    }
}
