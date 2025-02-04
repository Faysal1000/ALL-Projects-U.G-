namespace Faysals_Bluetooth_Communication_Project
{
    partial class firstPage
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(firstPage));
            txtbox = new TextBox();
            label1 = new Label();
            panel1 = new Panel();
            btnConnect = new Button();
            btnExit = new Button();
            label2 = new Label();
            linkLabel1 = new LinkLabel();
            SuspendLayout();
            // 
            // txtbox
            // 
            txtbox.BackColor = Color.Black;
            txtbox.BorderStyle = BorderStyle.None;
            txtbox.Font = new Font("Segoe UI", 8.25F, FontStyle.Regular, GraphicsUnit.Point, 0);
            txtbox.ForeColor = SystemColors.Info;
            txtbox.Location = new Point(94, 71);
            txtbox.MaxLength = 2;
            txtbox.Name = "txtbox";
            txtbox.PlaceholderText = "Enter valid integer number only";
            txtbox.Size = new Size(205, 15);
            txtbox.TabIndex = 0;
            txtbox.KeyPress += txtbox_KeyPress;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point, 0);
            label1.ForeColor = SystemColors.ButtonFace;
            label1.Location = new Point(94, 50);
            label1.Name = "label1";
            label1.Size = new Size(205, 18);
            label1.TabIndex = 1;
            label1.Text = "Enter Your COM number:";
            // 
            // panel1
            // 
            panel1.BackColor = Color.Cyan;
            panel1.Location = new Point(94, 86);
            panel1.Name = "panel1";
            panel1.Size = new Size(205, 5);
            panel1.TabIndex = 2;
            // 
            // btnConnect
            // 
            btnConnect.BackColor = Color.Transparent;
            btnConnect.Cursor = Cursors.Hand;
            btnConnect.FlatAppearance.BorderColor = Color.Green;
            btnConnect.FlatAppearance.MouseDownBackColor = Color.FromArgb(0, 192, 0);
            btnConnect.FlatAppearance.MouseOverBackColor = Color.FromArgb(128, 255, 128);
            btnConnect.FlatStyle = FlatStyle.Flat;
            btnConnect.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point, 0);
            btnConnect.ForeColor = SystemColors.ButtonFace;
            btnConnect.Location = new Point(94, 124);
            btnConnect.Name = "btnConnect";
            btnConnect.Size = new Size(97, 26);
            btnConnect.TabIndex = 3;
            btnConnect.Text = "Connect";
            btnConnect.UseVisualStyleBackColor = false;
            btnConnect.Click += btnConnect_Click;
            // 
            // btnExit
            // 
            btnExit.BackColor = Color.Transparent;
            btnExit.Cursor = Cursors.Hand;
            btnExit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            btnExit.FlatAppearance.MouseDownBackColor = Color.FromArgb(192, 0, 0);
            btnExit.FlatAppearance.MouseOverBackColor = Color.FromArgb(255, 128, 128);
            btnExit.FlatStyle = FlatStyle.Flat;
            btnExit.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point, 0);
            btnExit.ForeColor = SystemColors.ButtonFace;
            btnExit.Location = new Point(202, 124);
            btnExit.Name = "btnExit";
            btnExit.Size = new Size(97, 26);
            btnExit.TabIndex = 4;
            btnExit.Text = "Exit";
            btnExit.UseVisualStyleBackColor = false;
            btnExit.Click += btnExit_Click;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Segoe UI", 9F, FontStyle.Italic | FontStyle.Underline, GraphicsUnit.Point, 0);
            label2.ForeColor = Color.IndianRed;
            label2.Location = new Point(94, 153);
            label2.Name = "label2";
            label2.Size = new Size(212, 15);
            label2.TabIndex = 5;
            label2.Text = "Note: Only 9600 baud rate is supported";
            // 
            // linkLabel1
            // 
            linkLabel1.AutoSize = true;
            linkLabel1.BackColor = Color.Transparent;
            linkLabel1.Cursor = Cursors.Hand;
            linkLabel1.Location = new Point(181, 171);
            linkLabel1.Name = "linkLabel1";
            linkLabel1.Size = new Size(37, 15);
            linkLabel1.TabIndex = 6;
            linkLabel1.TabStop = true;
            linkLabel1.Text = "Help?";
            linkLabel1.LinkClicked += linkLabel1_LinkClicked;
            // 
            // firstPage
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(498, 221);
            Controls.Add(linkLabel1);
            Controls.Add(label2);
            Controls.Add(btnExit);
            Controls.Add(btnConnect);
            Controls.Add(panel1);
            Controls.Add(label1);
            Controls.Add(txtbox);
            DoubleBuffered = true;
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "firstPage";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "firstPage";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private TextBox txtbox;
        private Label label1;
        private Panel panel1;
        private Button btnConnect;
        private Button btnExit;
        private Label label2;
        private LinkLabel linkLabel1;
    }
}