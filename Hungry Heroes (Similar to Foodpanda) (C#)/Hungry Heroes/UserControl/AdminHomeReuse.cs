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
    public partial class AdminHomeReuse : UserControl, IAdminHomeReuse
    {
        public AdminHomeReuse()
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
        public void AdminHomeReuse_Load(object sender, EventArgs e)
        {
            int currentMonth = DateTime.Now.Month;
            double totalIncome = 0.0;
            double profit = 0.0;
            int totalUser = 0;
            int activeuser = 0;
            int totalResturant = 0;
            int activeResturant = 0;
            int totalguest = 0;
            int totalmanager = 0;
            //Database start
            SqlConnection con = databaseConnection();

            //now for total amount
            SqlCommand cmd = new SqlCommand("SELECT SUM(TOTAL) FROM BILL WHERE MONTH(BILLDATE) = @month", con);
            cmd.Parameters.AddWithValue("@month", currentMonth);
            try { totalIncome = double.Parse(cmd.ExecuteScalar().ToString()); } catch { totalIncome = 0.0; }
            profit = totalIncome * 0.1;

            //now for total user
            cmd = new SqlCommand("SELECT COUNT(*) FROM CUSTOMER", con);
            try { totalUser = int.Parse(cmd.ExecuteScalar().ToString()); } catch { totalUser = 0; }

            //now for active user
            cmd = new SqlCommand("SELECT COUNT(*) FROM CUSTOMER WHERE ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@status", "Active");
            try { activeuser = int.Parse(cmd.ExecuteScalar().ToString()); } catch { activeuser = 0; }

            //now for total resturant
            cmd = new SqlCommand("SELECT COUNT(*) FROM RESTURANT_OWNER", con);
            try { totalResturant = int.Parse(cmd.ExecuteScalar().ToString()); } catch { totalResturant = 0; }

            //now for active resturant
            cmd = new SqlCommand("SELECT COUNT(*) FROM RESTURANT_OWNER WHERE ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@status", "Active");
            try { activeResturant = int.Parse(cmd.ExecuteScalar().ToString()); } catch { activeResturant = 0; }

            //now for total guest user
            cmd = new SqlCommand("SELECT COUNT(*) FROM GUEST", con);
            try { totalguest = int.Parse(cmd.ExecuteScalar().ToString()); } catch { totalguest = 0; }

            //total manager currently working
            cmd = new SqlCommand("SELECT COUNT(*) FROM MANAGER WHERE ACCOUNT_STATUS=@status", con);
            cmd.Parameters.AddWithValue("@status", "Active");
            try { totalmanager = int.Parse(cmd.ExecuteScalar().ToString()); } catch { totalmanager = 0; }
            con.Close();
            //database end

            //now set values into label
            ltotal.Text = Math.Round(totalIncome, 2).ToString() + " TK";
            lprofit.Text = Math.Round(profit, 2).ToString() + " TK";
            ltotaluser.Text = totalUser.ToString();
            ltotalres.Text = totalResturant.ToString();
            lactiveuser.Text = activeuser.ToString();
            lactiveres.Text = activeResturant.ToString();
            lguestuser.Text = totalguest.ToString();
            lmanager.Text = totalmanager.ToString();
        }
    }
}
