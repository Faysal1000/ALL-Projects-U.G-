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
    public partial class ProfileReuse : UserControl
    {
        private CustomerHomePage customerHomePage;
        private string username;
        public ProfileReuse(CustomerHomePage x)
        {
            customerHomePage = x;
            InitializeComponent();
        }
        private SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        private void blogout_Click(object sender, EventArgs e)
        {
            DialogResult result = MessageBox.Show("Are you sure to logout?", "Confirmation",
                               MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                customerHomePage.Visible = false;
                CustomerLogin x = new CustomerLogin();
                x.Visible = true;
            }
        }

        public void profileReuseRefresh()
        {
            username = customerHomePage.username;
            string name;
            string email;
            string address;
            string birthdateString;
            DateTime birthday;

            //Database start
            SqlConnection con = databaseConnection();

            SqlCommand cmd = new SqlCommand("SELECT NAME FROM CUSTOMER WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
            cmd.Parameters.AddWithValue("@username", username);
            try { name = cmd.ExecuteScalar().ToString(); } catch { name = "Not Available"; }

            cmd = new SqlCommand("SELECT EMAIL FROM CUSTOMER WHERE USERNAME = @username", con);
            cmd.Parameters.AddWithValue("@username", username);
            try { email = cmd.ExecuteScalar().ToString(); } catch { email = "Not Available"; }

            cmd = new SqlCommand("SELECT ADDRESS FROM CUSTOMER WHERE USERNAME = @username", con);
            cmd.Parameters.AddWithValue("@username", username);
            try { address = cmd.ExecuteScalar().ToString(); } catch { address = "Not Available"; }

            cmd = new SqlCommand("SELECT BIRTHDATE FROM CUSTOMER WHERE USERNAME = @username", con);
            cmd.Parameters.AddWithValue("@username", username);
            try
            {
                birthday = (DateTime)cmd.ExecuteScalar();
                birthdateString = birthday.Date.ToString("yyyy-MM-dd");
            }
            catch { birthdateString = "Not Available"; }
            con.Close();
            //database end

            //setting value
            lsetname.Text = name;
            lsetusername.Text = username;
            lsetemail.Text = email;
            lsetaddress.Text = address;
            lsetbirth.Text = birthdateString;

        }

        private void ProfileReuse_Load(object sender, EventArgs e)
        {
            profileReuseRefresh();   
        }

        private void bupdate_Click(object sender, EventArgs e)
        {
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

             CustomerEdit x = new CustomerEdit(this,customerHomePage, name, username, email, address, birthday, sq, sqa, password);
             x.Visible = true;
        }
    }
}
