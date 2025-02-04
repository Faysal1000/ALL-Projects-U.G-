using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.DataFormats;

namespace Faysals_Bluetooth_Communication_Project
{
    public partial class firstPage : Form
    {
        public firstPage()
        {
            InitializeComponent();
            this.AcceptButton = btnConnect;
        }

        private void btnConnect_Click(object sender, EventArgs e)
        {
            String sCom = txtbox.Text.Trim();
            if (int.TryParse(sCom, out int parsedValue))
            {
                // Input is an integer, close the current form and navigate to another page
                this.Visible = false;
                SecondPage S = new SecondPage(parsedValue);
                S.Show();
            }
            else
            {
                MessageBox.Show("Please enter an integer.", "Invalid Input!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                txtbox.Text = string.Empty;
            }

        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void txtbox_KeyPress(object sender, KeyPressEventArgs e)
        {
            // Check if the pressed key is Enter
            if (e.KeyChar == (char)Keys.Enter)
            {
                // Call the click event handler of btnConnect
                btnConnect.PerformClick();
            }
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            this.Visible = false;
            helpPage h = new helpPage(this);
            h.Show();
        }
    }
}
