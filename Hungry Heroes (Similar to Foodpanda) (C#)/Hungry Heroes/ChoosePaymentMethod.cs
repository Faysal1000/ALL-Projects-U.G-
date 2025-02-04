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
    public partial class ChoosePaymentMethod : Form, IChoosePaymentMethod
    {
        CustomerOrder customerOrder;
        CustomerHomePage? customerHomePage;
        GuestHomePage? guestHomePage;
        private double payment = 0;
        private string resturantName;
        private double discount = 0;

        public ChoosePaymentMethod(CustomerOrder customerOrder, CustomerHomePage customerHomePage, string resturantName)
        {
            this.resturantName = resturantName;
            this.customerOrder = customerOrder;
            this.customerHomePage = customerHomePage;
            customerOrder.paymentMethodCheck = 1;
            InitializeComponent();
        }
        public ChoosePaymentMethod(CustomerOrder customerOrder, GuestHomePage guestHomePage, string resturantName)
        {
            this.payment = payment;
            this.resturantName = resturantName;
            this.customerOrder = customerOrder;
            this.guestHomePage = guestHomePage;
            customerOrder.paymentMethodCheck = 1;
            InitializeComponent();

            //for guest customer need to enter name and address
            lname.Visible = true;
            laddress.Visible = true;
            tbname.Visible = true;
            tbaddress.Visible = true;
            ptb1.Visible = true;
            ptb2.Visible = true;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void bbkash_Click(object sender, EventArgs e)
        {
            if (customerHomePage != null)
            {
                if (use.Checked)
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    BkashPaymentHome x = new BkashPaymentHome(customerOrder, customerHomePage, Math.Round(payment * discount, 2), resturantName,discount);
                    x.Visible = true;
                }
                else
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    BkashPaymentHome x = new BkashPaymentHome(customerOrder, customerHomePage, payment, resturantName, discount);
                    x.Visible = true;
                }

            }
            else
            {
                string name = tbname.Text;
                string address = tbaddress.Text;
                if (name == "" || address == "")
                {
                    MessageBox.Show("Please fill up all the fields first!", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    BkashPaymentHome x = new BkashPaymentHome(customerOrder, guestHomePage, Math.Round(payment, 2), resturantName, name, address);
                    x.Visible = true;
                }
            }
        }

        public void bmastercard_Click(object sender, EventArgs e)
        {
            if (customerHomePage != null)
            {
                if (use.Checked)
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    CardPayment x = new CardPayment(customerOrder, customerHomePage, Math.Round(payment * discount, 2), resturantName,discount);
                    x.Visible = true;
                }
                else
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    CardPayment x = new CardPayment(customerOrder, customerHomePage, payment, resturantName,discount);
                    x.Visible = true;
                }

            }
            else
            {
                string name = tbname.Text;
                string address = tbaddress.Text;
                if (name == "" || address == "")
                {
                    MessageBox.Show("Please fill up all the fields first!", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    this.Visible = false;
                    customerOrder.Visible = false;
                    CardPayment x = new CardPayment(customerOrder, guestHomePage, Math.Round(payment, 2), resturantName, name, address);
                    x.Visible = true;
                }

            }

        }

        public void bclose_Click(object sender, EventArgs e)
        {
            customerOrder.paymentMethodCheck = 0;
            this.Visible = false;
        }
        public void CheckOutDisplayLoad()
        {
            // Add columns to the DataGridView
            dataGridView.Columns.Add("FoodItem", "Food item");
            dataGridView.Columns.Add("Quantity", "Quantity");
            dataGridView.Columns.Add("Price", "Price");
            dataGridView.Columns.Add("Total", "Total");

            for (int i = 0; i < customerOrder.price.Count; i++)
            {
                double itemTotal = customerOrder.price[i] * customerOrder.quantity[i];
                payment += itemTotal;

                dataGridView.Rows.Add(customerOrder.foods[i], customerOrder.quantity[i], customerOrder.price[i], itemTotal);
            }
            Math.Round(payment, 2);
            // Add a row for the total
            dataGridView.Rows.Add("Total", "", "", payment);


            if (customerHomePage != null)
            {
                if (customerOrder.checkDiscount())
                {
                    //database start
                    SqlConnection con = databaseConnection();

                    //get discount
                    SqlCommand cmd = new SqlCommand("SELECT DISCOUNT FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname", con);
                    cmd.Parameters.AddWithValue("@resturantname", resturantName);
                    try { discount = int.Parse(cmd.ExecuteScalar().ToString()); } catch { discount = 0; }
                    con.Close();
                    //database end

                    ldiscount.Text = discount+"% Discount Available.Wanna Use it?";
                    ldiscount.Visible = true;
                    use.Visible = true;

                    double x = discount / 100;
                    discount = 1 - x;

                    dataGridView.Rows.Add("Discounted Price", "", "", Math.Round(payment * discount, 2));
                }
            }
            //select last one row
            dataGridView.Rows[dataGridView.Rows.Count - 2].Selected = true;
        }
        public void ChoosePaymentMethod_Load(object sender, EventArgs e)
        {
            CheckOutDisplayLoad();
        }

        public void button1_Click(object sender, EventArgs e)
        {
            if (dataGridView.SelectedCells.Count > 0)
            {
                int selectedRowIndex = dataGridView.SelectedCells[0].RowIndex;
                if(selectedRowIndex < customerOrder.foods.Count && customerOrder.foods.Count>=2)
                {
                    customerOrder.foods.RemoveAt(selectedRowIndex);
                    customerOrder.price.RemoveAt(selectedRowIndex);
                    customerOrder.quantity.RemoveAt(selectedRowIndex);

                    MessageBox.Show("Item Deleted!","Information", 
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    payment = 0.0;
                    dataGridView.Rows.Clear();
                    dataGridView.Columns.Clear();
                    CheckOutDisplayLoad();
                }
                else if(selectedRowIndex < customerOrder.foods.Count && customerOrder.foods.Count == 1)
                {
                    MessageBox.Show("Error Deleting! At least 1 Item Require to Checkout!",
                        "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                else
                {
                    MessageBox.Show("This Row Can't be Deleted!",
                        "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
            }
            else
            {
                MessageBox.Show("Select Something to delete!",
                        "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }
    }
}
