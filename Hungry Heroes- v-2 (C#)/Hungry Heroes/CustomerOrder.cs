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
    public partial class CustomerOrder : Form, ICustomerOrder
    {
        public int paymentMethodCheck { get; set; } = 0;
        private string resturantName;
        private string foodname;
        private double foodprice;
        public List<string> foods = new List<string>();
        public List<double> price = new List<double>();
        public List<int> quantity = new List<int>();
        public int maxItemCount = 0;

        CustomerHomePage? customerHomePage;
        GuestHomePage? guestHomePage;
        public CustomerOrder(CustomerHomePage x, string resturantName)
        {
            this.customerHomePage = x;
            InitializeComponent();
            this.resturantName = resturantName;
        }
        public CustomerOrder(GuestHomePage x, string resturantName)
        {
            this.guestHomePage = x;
            InitializeComponent();
            this.resturantName = resturantName;
        }

        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }

        public void baddfood_Click(object sender, EventArgs e)
        {
            if (paymentMethodCheck == 0)
            {
                if (display.SelectedRows.Count > 0)
                {
                    if (maxItemCount < 5)
                    {
                        foods.Add(foodname);
                        price.Add(foodprice);
                        quantity.Add(int.Parse(lquantitydisplay.Text));
                        maxItemCount++;
                        lquantitydisplay.Text = "1";
                        MessageBox.Show("Item Added Successfully!", "Information",
                                     MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        MessageBox.Show("You Can't order more than 5 different items at a time!", "Warning",
                                     MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
                else
                {
                    MessageBox.Show("Select an Item First!", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                }
            }
            else
            {
                MessageBox.Show("Close Previous Window First!", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
            }

        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void bprevious_Click(object sender, EventArgs e)
        {
            if (paymentMethodCheck == 0)
            {
                if (this.customerHomePage != null)
                {
                    this.Visible = false;
                    customerHomePage.Visible = true;
                }
                else
                {
                    this.Visible = false;
                    guestHomePage.Visible = true;
                }
            }
            else
            {
                MessageBox.Show("Close Previous Window First!", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
            }

        }

        public void bcheckout_Click(object sender, EventArgs e)
        {
            if (maxItemCount == 0)
            {
                MessageBox.Show("You haven't selected any item yet!", "Information",
                MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
            }
            else
            {
                if (customerHomePage != null)
                {
                    if (paymentMethodCheck == 0)
                    {
                        ChoosePaymentMethod x = new ChoosePaymentMethod(this, customerHomePage, resturantName);
                        x.Visible = true;
                    }
                    else
                    {
                        MessageBox.Show("Close Previous Window First!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                    }
                }
                else
                {
                    if (paymentMethodCheck == 0)
                    {
                        ChoosePaymentMethod x = new ChoosePaymentMethod(this, guestHomePage, resturantName);
                        x.Visible = true;
                    }
                    else
                    {
                        MessageBox.Show("Close Previous Window First!", "Information",
                            MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                    }
                }
            }

        }


        public void bplus_Click(object sender, EventArgs e)
        {
            if (paymentMethodCheck == 0)
            {
                int quantity = Convert.ToInt32(lquantitydisplay.Text);
                if (quantity < 10)
                {
                    quantity++;
                    lquantitydisplay.Text = quantity.ToString();
                }
                else
                {
                    MessageBox.Show("Can't Order More Than 10 Items at Once!", "Alert!",
                        MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                }
            }
            else
            {
                MessageBox.Show("Close Previous Window First!", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
            }

        }

        public void bminus_Click(object sender, EventArgs e)
        {

            if (paymentMethodCheck == 0)
            {
                int quantity = Convert.ToInt32(lquantitydisplay.Text);
                if (quantity > 1)
                {
                    quantity--;
                    lquantitydisplay.Text = quantity.ToString();
                }
                else
                {
                    MessageBox.Show("Can't Decrease Quantity Anymore!", "Alert!",
                        MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
                }
            }
            else
            {
                MessageBox.Show("Close Previous Window First!", "Information",
                    MessageBoxButtons.OK, MessageBoxIcon.Asterisk);
            }
        }

        public void CustomerOrder_Load(object sender, EventArgs e)
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT FOOD_NAME,FOOD_PRICE,FOOD_TYPE FROM MENU WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantName);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }

        public void display_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex >= 0 && e.RowIndex < display.Rows.Count)
            {
                DataGridViewRow row = display.Rows[e.RowIndex];
                foodname = row.Cells["FOOD_NAME"].Value.ToString();
                foodprice = double.Parse(row.Cells["FOOD_PRICE"].Value.ToString());
                ldisplay.Text = foodname + "\n\nPrice= " + foodprice;
            }
        }
        public bool checkDiscount()
        {
            int currentMonth = DateTime.Now.Month;
            double totalorder = 0.0;
            int check = 1;
            //Database start
            SqlConnection con = databaseConnection();

            SqlCommand cmd = new SqlCommand("SELECT SUM(TOTAL) FROM BILL WHERE MONTH(BILLDATE) = @month AND USERNAME=@username AND RESTURANT_NAME=@resturantname", con);
            cmd.Parameters.AddWithValue("@month", currentMonth);
            cmd.Parameters.AddWithValue("username", customerHomePage.username);
            cmd.Parameters.AddWithValue("@resturantname", resturantName);
            try { totalorder = double.Parse(cmd.ExecuteScalar().ToString()); } catch { totalorder = 0.0; }

            //CHECKING IF HE ALREADY USES HIS DISCOUNT OR NOT
            cmd = new SqlCommand("SELECT COUNT(DISCOUNT) FROM BILL WHERE MONTH(BILLDATE) = @month AND USERNAME=@username AND RESTURANT_NAME=@resturantname", con);
            cmd.Parameters.AddWithValue("@month", currentMonth);
            cmd.Parameters.AddWithValue("username", customerHomePage.username);
            cmd.Parameters.AddWithValue("@resturantname", resturantName);
            try { check = int.Parse(cmd.ExecuteScalar().ToString()); } catch { check = 1; }


            if (totalorder > 10000 && check == 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
