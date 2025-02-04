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
    public partial class FindResturantReuse : UserControl
    {
        CustomerHomePage? customerHomePage;
        GuestHomePage? guestHomePage;
        private string resturantName = "";
        public FindResturantReuse(CustomerHomePage x)
        {
            this.customerHomePage = x;
            InitializeComponent();
        }
        public FindResturantReuse(GuestHomePage x)
        {
            this.guestHomePage = x;
            InitializeComponent();
        }
        private SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        private void bshowmenu_Click(object sender, EventArgs e)
        {
            if (this.customerHomePage != null)
            {
                if (resturantName != "")
                {
                    customerHomePage.Visible = false;
                    CustomerOrder customerOrder = new CustomerOrder(customerHomePage, resturantName);
                    customerOrder.Visible = true;
                }
                else
                {
                    MessageBox.Show("Please Select a Resturant First!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                }

            }
            else
            {
                if (resturantName != "")
                {
                    guestHomePage.Visible = false;
                    CustomerOrder customerOrder = new CustomerOrder(guestHomePage, resturantName);
                    customerOrder.Visible = true;
                }
                else
                {
                    MessageBox.Show("Please Select a Resturant First!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                }

            }

        }

        private void FindResturantReuse_Load(object sender, EventArgs e)
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT RESTURANT_NAME FROM RESTURANT_OWNER WHERE AVAILABILITY = @yesORno AND ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@yesORno", "YES");
            cmd.Parameters.AddWithValue("@status", "Active");
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        private void display_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex >= 0 && e.RowIndex < display.Rows.Count)
            {
                DataGridViewRow row = display.Rows[e.RowIndex];
                resturantName = row.Cells["RESTURANT_NAME"].Value.ToString();
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (tbsearch.Text == "")
            {
                MessageBox.Show("Write Resturnat Name to search!", "Information!",
                    MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
            else
            {
                //Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT RESTURANT_NAME FROM RESTURANT_OWNER WHERE AVAILABILITY = @yesORno AND ACCOUNT_STATUS=@status AND RESTURANT_NAME= @resturantname", con);
                cmd.Parameters.AddWithValue("@yesORno", "YES");
                cmd.Parameters.AddWithValue("@status", "Active");
                cmd.Parameters.AddWithValue("@resturantname", tbsearch.Text);
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                DataTable dt = new DataTable();
                da.Fill(dt);
                con.Close();
                if (dt.Rows.Count > 0)
                {
                    display.DataSource = dt;
                    display.ClearSelection();
                }
                else
                {
                    MessageBox.Show("Resturant Not Found!", "Information!",
                    MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                }
                //database end here
            }
        }

        private void bsearch_Click(object sender, EventArgs e)
        {
            tbsearch.Text = string.Empty;
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT RESTURANT_NAME FROM RESTURANT_OWNER WHERE AVAILABILITY = @yesORno AND ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@yesORno", "YES");
            cmd.Parameters.AddWithValue("@status", "Active");
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
