using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Hungry_Heroes
{
    public partial class PaymentSuccess : Form, IPaymentSuccess
    {
        CustomerHomePage? customerHomePage;
        GuestHomePage? guestHomePage;
        double payment = 0;
        private string resturantName;
        public int ratingdone { get; set; } = 0;
        public int ratingcheck { get; set; } = 0;
        public PaymentSuccess(CustomerHomePage customerHomePage, double payment, string resturantName)
        {
            this.payment = payment;
            this.customerHomePage = customerHomePage;
            InitializeComponent();
            lamount.Text = "Amount Paid: " + payment + " TK";
            this.resturantName = resturantName;
        }
        public PaymentSuccess(GuestHomePage guestHomePage, double payment, string resturantName)
        {
            this.payment = payment;
            this.guestHomePage = guestHomePage;
            InitializeComponent();
            lamount.Text = "Amount Paid: " + payment + " TK";
            brate.Visible = false;
            this.resturantName = resturantName;
        }

        public void brate_Click(object sender, EventArgs e)
        {
            if (ratingcheck == 0 && ratingdone == 0)
            {
                Rating x = new Rating(this, resturantName);
                x.Visible = true;
            }
            else if (ratingdone != 0)
            {
                MessageBox.Show("You Already Rated this Resturant!", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                MessageBox.Show("Close Previous Windows first", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void bhome_Click(object sender, EventArgs e)
        {
            if (ratingcheck == 0)
            {
                if (customerHomePage != null)
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
                MessageBox.Show("Close Previous Windows first", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }
    }
}
