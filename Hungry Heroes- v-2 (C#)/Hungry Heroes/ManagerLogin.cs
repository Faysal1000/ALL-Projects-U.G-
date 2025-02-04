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
    public partial class ManagerLogin : Form, IManagerLogin
    {
        public ManagerLogin()
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
        public void beyeok_Click(object sender, EventArgs e)
        {
            beyecross.Visible = true;
            beyeok.Visible = false;
            tbpass.UseSystemPasswordChar = false;
        }

        public void beyecross_Click(object sender, EventArgs e)
        {
            beyecross.Visible = false;
            beyeok.Visible = true;
            tbpass.UseSystemPasswordChar = true;
        }

        public void blogin_Click(object sender, EventArgs e)
        {
            string username = tbusername.Text;
            string pass = tbpass.Text;
            string sans = tbcans.Text;
            int ans = 0;

            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT * FROM MANAGER WHERE USERNAME = @username COLLATE Latin1_General_BIN AND PASSWORD = @password COLLATE Latin1_General_BIN AND ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@username", username);
            cmd.Parameters.AddWithValue("@password", pass);
            cmd.Parameters.AddWithValue("@status", "Active");

            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);

            if (tbcans.Text != "")
            {
                try
                {
                    ans = int.Parse(sans);
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


            if (username == "" || pass == "")
            {
                MessageBox.Show("Please fill up all the fields", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else if (dt.Rows.Count > 0 && check != ans)
            {
                if (ans != -1)
                {
                    MessageBox.Show("Capta Ans. is wrong", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            else if (dt.Rows.Count > 0 && check == ans)
            {
                //now for login details table
                string query = "INSERT INTO LOGIN_DETAILS(TIME,TYPE, USERNAME) VALUES (@time, @type,@username)";
                cmd = new SqlCommand(query, con);
                cmd.Parameters.AddWithValue("@time", DateTime.Now);
                cmd.Parameters.AddWithValue("@type", "MANAGER");
                cmd.Parameters.AddWithValue("@username", username);
                cmd.ExecuteNonQuery();
                con.Close();
                //database end here

                this.Visible = false;
                ManagerHomePage x = new ManagerHomePage();
                x.Visible = true;
            }
            else
            {
                MessageBox.Show("Invalid Username or Password", "Invalid Input", MessageBoxButtons.OK, MessageBoxIcon.Error);
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
    }
}
