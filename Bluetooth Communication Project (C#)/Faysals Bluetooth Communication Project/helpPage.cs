using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Faysals_Bluetooth_Communication_Project
{
    public partial class helpPage : Form
    {
        firstPage? f;
        public helpPage(firstPage f)
        {
            this.f = f;
            InitializeComponent();
        }

        private void btnConnect_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            f.Visible= true;
        }
    }
}
