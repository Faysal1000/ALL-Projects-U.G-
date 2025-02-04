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
using System.Xml.Linq;

namespace Hungry_Heroes
{
    public partial class ForgotPasswordHome : Form, IForgotPasswordHome
    {
        public ForgotPasswordHome()
        {
            InitializeComponent();
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void bprevious_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            CustomerLogin x = new CustomerLogin();
            x.Visible = true;
        }

        public void bgetpass_Click(object sender, EventArgs e)
        {
            //value picking
            string username = tbusername.Text;
            string sq = cbsq.Text;
            string sqa = tbsqa.Text;


            if (username == "" || sq == "" || sqa == "")
            {
                MessageBox.Show("Please fill up all the fields", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                //searching username. if found then assign it to getusername else leave it as it is
                //Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
                cmd.Parameters.AddWithValue("@username", username);

                SqlDataAdapter da = new SqlDataAdapter(cmd);
                DataTable dt = new DataTable();
                da.Fill(dt);
                //......


                if (dt.Rows.Count > 0)
                {
                    //retrive security question and its ans for the username
                    cmd = new SqlCommand("SELECT * FROM CUSTOMER WHERE SQ = @sq COLLATE Latin1_General_BIN AND SQA = @sqa COLLATE Latin1_General_BIN", con);
                    cmd.Parameters.AddWithValue("@sq", sq);
                    cmd.Parameters.AddWithValue("@sqa", sqa);
                    da = new SqlDataAdapter(cmd);
                    DataTable dt2 = new DataTable();
                    da.Fill(dt2);
                    //.....


                    if (dt2.Rows.Count>0)
                    {
                        //get Database pass for the username
                        cmd = new SqlCommand("SELECT PASSWORD FROM CUSTOMER WHERE USERNAME = @username", con);
                        cmd.Parameters.AddWithValue("@username", username);
                        string databasePass = cmd.ExecuteScalar().ToString();
                        con.Close();
                        //database ended

                        MessageBox.Show("Your Password is: " + databasePass, "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);

                        //clearing fields
                        tbusername.Text = string.Empty;
                        tbsqa.Text = string.Empty;
                        cbsq.SelectedIndex = -1;
                    }
                    else
                    {
                        MessageBox.Show("Wrong Sequrity Question or Answer ", "Error",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }

                }
                else
                {
                    MessageBox.Show("User Not Found", "Error",
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
            }
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }
    }
}
