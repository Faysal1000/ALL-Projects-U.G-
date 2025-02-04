namespace Hungry_Heroes
{
    partial class CardPayment
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CardPayment));
            label1 = new Label();
            label2 = new Label();
            pictureBox1 = new PictureBox();
            label3 = new Label();
            label4 = new Label();
            label5 = new Label();
            label6 = new Label();
            label7 = new Label();
            tbname = new TextBox();
            tbcardno = new TextBox();
            tbvalid = new TextBox();
            tbcode = new TextBox();
            bprevious = new Button();
            blogin = new Button();
            pictureBox3 = new PictureBox();
            beyeok = new Button();
            beyecross = new Button();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBox3).BeginInit();
            SuspendLayout();
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Cooper Black", 18F, FontStyle.Regular, GraphicsUnit.Point);
            label1.ForeColor = Color.Black;
            label1.ImageAlign = ContentAlignment.MiddleLeft;
            label1.Location = new Point(404, 18);
            label1.Name = "label1";
            label1.Size = new Size(310, 27);
            label1.TabIndex = 83;
            label1.Text = "Complete Your Payment";
            label1.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Georgia", 14.25F, FontStyle.Bold, GraphicsUnit.Point);
            label2.ForeColor = Color.Black;
            label2.ImageAlign = ContentAlignment.MiddleLeft;
            label2.Location = new Point(405, 67);
            label2.Name = "label2";
            label2.Size = new Size(165, 23);
            label2.TabIndex = 84;
            label2.Text = "We Accept Only";
            label2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // pictureBox1
            // 
            pictureBox1.Anchor = AnchorStyles.None;
            pictureBox1.Image = Properties.Resources.wao11;
            pictureBox1.Location = new Point(567, 64);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(54, 34);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 85;
            pictureBox1.TabStop = false;
            // 
            // label3
            // 
            label3.Anchor = AnchorStyles.None;
            label3.AutoSize = true;
            label3.BackColor = Color.Transparent;
            label3.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point);
            label3.ForeColor = Color.DimGray;
            label3.ImageAlign = ContentAlignment.MiddleLeft;
            label3.Location = new Point(405, 112);
            label3.Name = "label3";
            label3.Size = new Size(314, 18);
            label3.TabIndex = 87;
            label3.Text = "Please Fillup Your Card Details Below--";
            label3.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label4
            // 
            label4.Anchor = AnchorStyles.None;
            label4.AutoSize = true;
            label4.BackColor = Color.Transparent;
            label4.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label4.ForeColor = Color.DimGray;
            label4.ImageAlign = ContentAlignment.MiddleLeft;
            label4.Location = new Point(404, 141);
            label4.Name = "label4";
            label4.Size = new Size(121, 17);
            label4.TabIndex = 88;
            label4.Text = "Name on Card: ";
            label4.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label5
            // 
            label5.Anchor = AnchorStyles.None;
            label5.AutoSize = true;
            label5.BackColor = Color.Transparent;
            label5.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label5.ForeColor = Color.DimGray;
            label5.ImageAlign = ContentAlignment.MiddleLeft;
            label5.Location = new Point(404, 179);
            label5.Name = "label5";
            label5.Size = new Size(111, 17);
            label5.TabIndex = 89;
            label5.Text = "Card Number:";
            label5.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label6
            // 
            label6.Anchor = AnchorStyles.None;
            label6.AutoSize = true;
            label6.BackColor = Color.Transparent;
            label6.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label6.ForeColor = Color.DimGray;
            label6.ImageAlign = ContentAlignment.MiddleLeft;
            label6.Location = new Point(404, 220);
            label6.Name = "label6";
            label6.Size = new Size(71, 17);
            label6.TabIndex = 90;
            label6.Text = "Valid on:";
            label6.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // label7
            // 
            label7.Anchor = AnchorStyles.None;
            label7.AutoSize = true;
            label7.BackColor = Color.Transparent;
            label7.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label7.ForeColor = Color.DimGray;
            label7.ImageAlign = ContentAlignment.MiddleLeft;
            label7.Location = new Point(589, 220);
            label7.Name = "label7";
            label7.Size = new Size(90, 17);
            label7.TabIndex = 91;
            label7.Text = " CVV Code:";
            label7.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.Control;
            tbname.BorderStyle = BorderStyle.FixedSingle;
            tbname.Cursor = Cursors.IBeam;
            tbname.Location = new Point(521, 139);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Enter Name";
            tbname.Size = new Size(246, 23);
            tbname.TabIndex = 92;
            // 
            // tbcardno
            // 
            tbcardno.Anchor = AnchorStyles.None;
            tbcardno.BackColor = SystemColors.Control;
            tbcardno.BorderStyle = BorderStyle.FixedSingle;
            tbcardno.Cursor = Cursors.IBeam;
            tbcardno.Location = new Point(521, 178);
            tbcardno.MaxLength = 20;
            tbcardno.Name = "tbcardno";
            tbcardno.PlaceholderText = "Enter Card Number";
            tbcardno.Size = new Size(246, 23);
            tbcardno.TabIndex = 93;
            // 
            // tbvalid
            // 
            tbvalid.Anchor = AnchorStyles.None;
            tbvalid.BackColor = SystemColors.Control;
            tbvalid.BorderStyle = BorderStyle.FixedSingle;
            tbvalid.Cursor = Cursors.IBeam;
            tbvalid.Location = new Point(473, 218);
            tbvalid.MaxLength = 15;
            tbvalid.Name = "tbvalid";
            tbvalid.PlaceholderText = "Validity date";
            tbvalid.Size = new Size(119, 23);
            tbvalid.TabIndex = 94;
            // 
            // tbcode
            // 
            tbcode.Anchor = AnchorStyles.None;
            tbcode.BackColor = SystemColors.Control;
            tbcode.BorderStyle = BorderStyle.FixedSingle;
            tbcode.Cursor = Cursors.IBeam;
            tbcode.Location = new Point(677, 218);
            tbcode.MaxLength = 3;
            tbcode.Name = "tbcode";
            tbcode.PlaceholderText = "Enter Code";
            tbcode.Size = new Size(67, 23);
            tbcode.TabIndex = 95;
            tbcode.UseSystemPasswordChar = true;
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
            bprevious.Location = new Point(403, 305);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(363, 34);
            bprevious.TabIndex = 97;
            bprevious.Text = "Cancel Order";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // blogin
            // 
            blogin.Anchor = AnchorStyles.None;
            blogin.BackColor = Color.Transparent;
            blogin.BackgroundImageLayout = ImageLayout.Stretch;
            blogin.Cursor = Cursors.Hand;
            blogin.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            blogin.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            blogin.FlatAppearance.MouseOverBackColor = Color.Lime;
            blogin.FlatStyle = FlatStyle.Flat;
            blogin.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            blogin.ForeColor = Color.Black;
            blogin.Image = (Image)resources.GetObject("blogin.Image");
            blogin.ImageAlign = ContentAlignment.MiddleLeft;
            blogin.Location = new Point(404, 264);
            blogin.Name = "blogin";
            blogin.Size = new Size(362, 34);
            blogin.TabIndex = 96;
            blogin.Text = "Pay";
            blogin.UseVisualStyleBackColor = false;
            blogin.Click += blogin_Click;
            // 
            // pictureBox3
            // 
            pictureBox3.Anchor = AnchorStyles.None;
            pictureBox3.Image = Properties.Resources.f754ef05e2e46234a2f8f60b5ee80041;
            pictureBox3.Location = new Point(2, 3);
            pictureBox3.Name = "pictureBox3";
            pictureBox3.Size = new Size(395, 341);
            pictureBox3.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox3.TabIndex = 98;
            pictureBox3.TabStop = false;
            // 
            // beyeok
            // 
            beyeok.Anchor = AnchorStyles.None;
            beyeok.BackColor = SystemColors.ScrollBar;
            beyeok.BackgroundImageLayout = ImageLayout.Stretch;
            beyeok.Cursor = Cursors.Hand;
            beyeok.FlatStyle = FlatStyle.Popup;
            beyeok.Image = (Image)resources.GetObject("beyeok.Image");
            beyeok.Location = new Point(742, 218);
            beyeok.Name = "beyeok";
            beyeok.Size = new Size(25, 23);
            beyeok.TabIndex = 100;
            beyeok.UseMnemonic = false;
            beyeok.UseVisualStyleBackColor = false;
            beyeok.Click += beyeok_Click;
            // 
            // beyecross
            // 
            beyecross.Anchor = AnchorStyles.None;
            beyecross.BackColor = SystemColors.ScrollBar;
            beyecross.BackgroundImageLayout = ImageLayout.Stretch;
            beyecross.Cursor = Cursors.Hand;
            beyecross.FlatStyle = FlatStyle.Popup;
            beyecross.Image = Properties.Resources.eyecross;
            beyecross.Location = new Point(742, 218);
            beyecross.Name = "beyecross";
            beyecross.Size = new Size(25, 23);
            beyecross.TabIndex = 99;
            beyecross.UseMnemonic = false;
            beyecross.UseVisualStyleBackColor = false;
            beyecross.Visible = false;
            beyecross.Click += beyecross_Click;
            // 
            // CardPayment
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(784, 369);
            Controls.Add(beyeok);
            Controls.Add(beyecross);
            Controls.Add(pictureBox3);
            Controls.Add(bprevious);
            Controls.Add(blogin);
            Controls.Add(tbcode);
            Controls.Add(tbvalid);
            Controls.Add(tbcardno);
            Controls.Add(tbname);
            Controls.Add(label7);
            Controls.Add(label6);
            Controls.Add(label5);
            Controls.Add(label4);
            Controls.Add(label3);
            Controls.Add(pictureBox1);
            Controls.Add(label2);
            Controls.Add(label1);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "CardPayment";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "CardPayment";
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBox3).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label label1;
        private Label label2;
        private PictureBox pictureBox1;
        private Label label3;
        private Label label4;
        private Label label5;
        private Label label6;
        private Label label7;
        private TextBox tbname;
        private TextBox tbcardno;
        private TextBox tbvalid;
        private TextBox tbcode;
        private Button bprevious;
        private Button blogin;
        private PictureBox pictureBox3;
        private Button beyeok;
        private Button beyecross;
    }
}