namespace Hungry_Heroes
{
    partial class BkashPaymentHome
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(BkashPaymentHome));
            bprevious = new Button();
            bpay = new Button();
            pictureBox1 = new PictureBox();
            label1 = new Label();
            label2 = new Label();
            ptb3 = new Panel();
            lfoodprice = new Label();
            tbno = new TextBox();
            panel1 = new Panel();
            label3 = new Label();
            tbpin = new TextBox();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.BackColor = Color.Transparent;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bprevious.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bprevious.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.ForeColor = Color.Black;
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(102, 393);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(244, 34);
            bprevious.TabIndex = 79;
            bprevious.Text = "Cancel Order";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // bpay
            // 
            bpay.Anchor = AnchorStyles.None;
            bpay.BackColor = Color.Transparent;
            bpay.BackgroundImageLayout = ImageLayout.Stretch;
            bpay.Cursor = Cursors.Hand;
            bpay.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bpay.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bpay.FlatAppearance.MouseOverBackColor = Color.Lime;
            bpay.FlatStyle = FlatStyle.Flat;
            bpay.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bpay.ForeColor = Color.Black;
            bpay.Image = (Image)resources.GetObject("bpay.Image");
            bpay.ImageAlign = ContentAlignment.MiddleLeft;
            bpay.Location = new Point(103, 353);
            bpay.Name = "bpay";
            bpay.Size = new Size(243, 34);
            bpay.TabIndex = 78;
            bpay.Text = "Pay";
            bpay.UseVisualStyleBackColor = false;
            bpay.Click += blogin_Click;
            // 
            // pictureBox1
            // 
            pictureBox1.Anchor = AnchorStyles.None;
            pictureBox1.Image = Properties.Resources.bkash_logo_835789094A_seeklogo_com;
            pictureBox1.Location = new Point(101, 12);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(245, 138);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 81;
            pictureBox1.TabStop = false;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Franklin Gothic Book", 14.25F, FontStyle.Bold, GraphicsUnit.Point);
            label1.ForeColor = Color.Black;
            label1.ImageAlign = ContentAlignment.MiddleLeft;
            label1.Location = new Point(97, 186);
            label1.Name = "label1";
            label1.Size = new Size(64, 24);
            label1.TabIndex = 82;
            label1.Text = "Log In";
            label1.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label2.ForeColor = Color.Black;
            label2.ImageAlign = ContentAlignment.MiddleLeft;
            label2.Location = new Point(98, 210);
            label2.Name = "label2";
            label2.Size = new Size(152, 20);
            label2.TabIndex = 83;
            label2.Text = "to your bKash account";
            label2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // ptb3
            // 
            ptb3.Anchor = AnchorStyles.None;
            ptb3.BackColor = SystemColors.MenuHighlight;
            ptb3.Location = new Point(104, 267);
            ptb3.Name = "ptb3";
            ptb3.Size = new Size(246, 2);
            ptb3.TabIndex = 86;
            // 
            // lfoodprice
            // 
            lfoodprice.Anchor = AnchorStyles.None;
            lfoodprice.AutoSize = true;
            lfoodprice.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            lfoodprice.Location = new Point(101, 237);
            lfoodprice.Name = "lfoodprice";
            lfoodprice.Size = new Size(105, 14);
            lfoodprice.TabIndex = 85;
            lfoodprice.Text = "Account Number";
            // 
            // tbno
            // 
            tbno.Anchor = AnchorStyles.None;
            tbno.BackColor = SystemColors.Control;
            tbno.BorderStyle = BorderStyle.None;
            tbno.Cursor = Cursors.IBeam;
            tbno.Location = new Point(103, 251);
            tbno.MaxLength = 15;
            tbno.Name = "tbno";
            tbno.PlaceholderText = "Enter Account Number";
            tbno.Size = new Size(246, 16);
            tbno.TabIndex = 84;
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.None;
            panel1.BackColor = SystemColors.MenuHighlight;
            panel1.Location = new Point(104, 311);
            panel1.Name = "panel1";
            panel1.Size = new Size(246, 2);
            panel1.TabIndex = 89;
            // 
            // label3
            // 
            label3.Anchor = AnchorStyles.None;
            label3.AutoSize = true;
            label3.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            label3.Location = new Point(102, 281);
            label3.Name = "label3";
            label3.Size = new Size(66, 14);
            label3.TabIndex = 88;
            label3.Text = "bKash Pin";
            // 
            // tbpin
            // 
            tbpin.Anchor = AnchorStyles.None;
            tbpin.BackColor = SystemColors.Control;
            tbpin.BorderStyle = BorderStyle.None;
            tbpin.Cursor = Cursors.IBeam;
            tbpin.Location = new Point(103, 295);
            tbpin.MaxLength = 10;
            tbpin.Name = "tbpin";
            tbpin.PlaceholderText = "Enter bKash Pin";
            tbpin.Size = new Size(246, 16);
            tbpin.TabIndex = 87;
            tbpin.UseSystemPasswordChar = true;
            // 
            // BkashPaymentHome
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(423, 478);
            Controls.Add(panel1);
            Controls.Add(label3);
            Controls.Add(tbpin);
            Controls.Add(ptb3);
            Controls.Add(lfoodprice);
            Controls.Add(tbno);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(pictureBox1);
            Controls.Add(bprevious);
            Controls.Add(bpay);
            FormBorderStyle = FormBorderStyle.Fixed3D;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "BkashPaymentHome";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "BkashPaymentHome";
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private Button bprevious;
        private Button bpay;
        private PictureBox pictureBox1;
        private Label label1;
        private Label label2;
        private Panel ptb3;
        private Label lfoodprice;
        private TextBox tbno;
        private Panel panel1;
        private Label label3;
        private TextBox tbpin;
    }
}