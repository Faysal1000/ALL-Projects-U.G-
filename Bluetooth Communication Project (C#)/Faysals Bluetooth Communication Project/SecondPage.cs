using System;
using System.Diagnostics;
using System.Windows.Forms;
using RJCP.IO.Ports;
using static System.Windows.Forms.DataFormats;

namespace Faysals_Bluetooth_Communication_Project
{
    public partial class SecondPage : Form
    {
        SerialPortStream? serialPort;
        int portNo;
        public SecondPage(int portNo)
        {
            this.portNo = portNo;
            InitializeComponent();
            InitializeSerialPort();

        }
        private void InitializeSerialPort()
        {
            string portName = "COM" + portNo;
            //need to change com number accordingly
            serialPort = new SerialPortStream(portName, 9600);

            try
            {
                serialPort.Open();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error opening serial port: " + ex.Message, "Error.!!!!", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            if (serialPort.IsOpen)
            {
                lconnected.Visible = true;
                lconnected.Text = portName + " Connected";
                lerror.Visible = false;
                btnPrevious.Visible = false;
                btnRetry.Visible = false;
            }
        }
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            // Close the serial port when the form is closing
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Close();
            }

            base.OnFormClosing(e);
        }
        private void front_MouseDown(object sender, MouseEventArgs e)
        {
            front.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'f' }, 0, 1);
            }
        }

        private void back_MouseDown(object sender, MouseEventArgs e)
        {
            back.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'b' }, 0, 1);
            }
        }

        private void rightt_MouseDown(object sender, MouseEventArgs e)
        {
            rightt.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'r' }, 0, 1);
            }
        }

        private void leftt_MouseDown(object sender, MouseEventArgs e)
        {
            leftt.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'l' }, 0, 1);
            }
        }

        private void up_MouseDown(object sender, MouseEventArgs e)
        {
            up.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'u' }, 0, 1);
            }
        }

        private void down_MouseDown(object sender, MouseEventArgs e)
        {
            down.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'d' }, 0, 1);
            }
        }

        private void open_MouseDown(object sender, MouseEventArgs e)
        {
            open.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'o' }, 0, 1);
            }
        }

        private void close_MouseDown(object sender, MouseEventArgs e)
        {
            close.BackColor = Color.Red;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'c' }, 0, 1);
            }
        }

        private void close_MouseUp(object sender, MouseEventArgs e)
        {
            front.BackColor = Color.Transparent;
            back.BackColor = Color.Transparent;
            leftt.BackColor = Color.Transparent;
            rightt.BackColor = Color.Transparent;
            up.BackColor = Color.Transparent;
            down.BackColor = Color.Transparent;
            open.BackColor = Color.Transparent;
            close.BackColor = Color.Transparent;

            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'x' }, 0, 1);
            }
        }

        private void forward_MouseDown(object sender, MouseEventArgs e)
        {
            forward.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'F' }, 0, 1);
            }
        }

        private void backward_MouseDown(object sender, MouseEventArgs e)
        {
            backward.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'B' }, 0, 1);
            }
        }

        private void right_MouseDown(object sender, MouseEventArgs e)
        {
            right.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'R' }, 0, 1);
            }
        }

        private void left_MouseDown(object sender, MouseEventArgs e)
        {
            left.BackColor = Color.Red;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'L' }, 0, 1);
            }
        }

        private void left_MouseUp(object sender, MouseEventArgs e)
        {
            forward.BackColor = Color.Transparent;
            backward.BackColor = Color.Transparent;
            left.BackColor = Color.Transparent;
            right.BackColor = Color.Transparent;
            if (serialPort != null && serialPort.IsOpen)
            {
                serialPort.Write(new byte[] { (byte)'S' }, 0, 1);
            }
        }

        private void label3_Click(object sender, EventArgs e)
        {
            string facebookProfileUsername = "faysal.ahmmed.2001";
            string facebookProfileUrl = "https://www.facebook.com/" + facebookProfileUsername;

            // Specify the process start information
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = facebookProfileUrl,
                UseShellExecute = true // Use the system's default shell to open the URL
            };

            // Start the process
            try
            {
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error opening Facebook profile: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void label3_MouseEnter(object sender, EventArgs e)
        {
            label3.ForeColor = Color.Red;
        }

        private void label3_MouseLeave(object sender, EventArgs e)
        {
            label3.ForeColor = Color.Black;

        }

        private void up_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Up"; 
        }

        private void front_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Front";
        }

        private void rightt_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Right";
        }

        private void leftt_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Left";
        }

        private void back_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Back";
        }

        private void open_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Claw-Open";
        }

        private void close_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Claw-Close";
        }

        private void down_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Arm-Down";
        }

        private void forward_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Car-Forward";
        }

        private void right_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Car-Right";
        }

        private void backward_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Car-Back";
        }

        private void left_MouseEnter(object sender, EventArgs e)
        {
            ldisplay.Visible = true;
            ldisplay.Text = "Car-Left";
        }

        private void left_MouseLeave(object sender, EventArgs e)
        {
            ldisplay.Visible = false;
        }

        private void btnRetry_Click(object sender, EventArgs e)
        {
            InitializeSerialPort();
        }

        private void btnPrevious_Click(object sender, EventArgs e)
        {
            firstPage f = new firstPage();
            this.Visible=false;
            f.Show();
        }
    }
}
