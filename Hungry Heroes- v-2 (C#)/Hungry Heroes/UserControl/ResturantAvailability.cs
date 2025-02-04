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
    public partial class ResturantAvailability : UserControl
    {
        private string resturantName;
        public ResturantAvailability(string resturantName)
        {
            InitializeComponent();
            this.resturantName = resturantName;
        }
        private SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        private void bstatus_Click(object sender, EventArgs e)
        {
            if (available.Checked)
            {
                //Database Start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET AVAILABILITY=@status WHERE RESTURANT_NAME=@resturantname", con);
                cmd.Parameters.AddWithValue("@status", "Yes");
                cmd.Parameters.AddWithValue("@resturantname", resturantName);

                cmd.ExecuteNonQuery();
                con.Close();
                //Database closed
                statuspic.Visible = false;
                MessageBox.Show("Status Updated!","Successfull!",MessageBoxButtons.OK,MessageBoxIcon.Information);
            }
            else
            {
                //Database Start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET AVAILABILITY=@status WHERE RESTURANT_NAME=@resturantname", con);
                cmd.Parameters.AddWithValue("@status", "No");
                cmd.Parameters.AddWithValue("@resturantname", resturantName);

                cmd.ExecuteNonQuery();
                con.Close();
                //Database closed
                statuspic.Visible = true;
                MessageBox.Show("Status Updated!", "Successfull!", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

        }

        private void ResturantAvailability_Load(object sender, EventArgs e)
        {
            //setting previous status
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT AVAILABILITY FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantName);
            string status = cmd.ExecuteScalar().ToString();
            con.Close();
            //database ended
            if (status.Equals("YES", StringComparison.OrdinalIgnoreCase))
            {
                available.Checked= true;
                statuspic.Visible = false;
            }
            else
            {
                notavailable.Checked = true;
                statuspic.Visible = true;
            }
        }
    }
}
