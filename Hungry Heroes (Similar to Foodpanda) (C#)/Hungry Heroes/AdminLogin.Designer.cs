namespace Hungry_Heroes
{
    partial class AdminLogin
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AdminLogin));
            panel1 = new Panel();
            pictureBox1 = new PictureBox();
            lcaptcha = new Label();
            tbcans = new TextBox();
            lcpequal = new Label();
            lcp2 = new Label();
            lcpplus = new Label();
            lcp1 = new Label();
            bexit = new Button();
            bprevious = new Button();
            blogin = new Button();
            beyeok = new Button();
            beyecross = new Button();
            tbpass = new TextBox();
            tbusername = new TextBox();
            lpass = new Label();
            lusername = new Label();
            brecapta = new Button();
            panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.None;
            panel1.BackColor = Color.Transparent;
            panel1.BorderStyle = BorderStyle.Fixed3D;
            panel1.Controls.Add(brecapta);
            panel1.Controls.Add(pictureBox1);
            panel1.Controls.Add(lcaptcha);
            panel1.Controls.Add(tbcans);
            panel1.Controls.Add(lcpequal);
            panel1.Controls.Add(lcp2);
            panel1.Controls.Add(lcpplus);
            panel1.Controls.Add(lcp1);
            panel1.Controls.Add(bexit);
            panel1.Controls.Add(bprevious);
            panel1.Controls.Add(blogin);
            panel1.Controls.Add(beyeok);
            panel1.Controls.Add(beyecross);
            panel1.Controls.Add(tbpass);
            panel1.Controls.Add(tbusername);
            panel1.Controls.Add(lpass);
            panel1.Controls.Add(lusername);
            panel1.Location = new Point(251, 67);
            panel1.Name = "panel1";
            panel1.Size = new Size(323, 358);
            panel1.TabIndex = 1;
            // 
            // pictureBox1
            // 
            pictureBox1.BackColor = Color.Transparent;
            pictureBox1.Image = Properties.Resources.giphy1;
            pictureBox1.Location = new Point(138, 3);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(76, 70);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 80;
            pictureBox1.TabStop = false;
            // 
            // lcaptcha
            // 
            lcaptcha.Anchor = AnchorStyles.None;
            lcaptcha.AutoSize = true;
            lcaptcha.BackColor = Color.Transparent;
            lcaptcha.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcaptcha.ForeColor = Color.Gainsboro;
            lcaptcha.Image = (Image)resources.GetObject("lcaptcha.Image");
            lcaptcha.ImageAlign = ContentAlignment.MiddleLeft;
            lcaptcha.Location = new Point(12, 158);
            lcaptcha.Name = "lcaptcha";
            lcaptcha.Size = new Size(82, 20);
            lcaptcha.TabIndex = 79;
            lcaptcha.Text = "     Captcha";
            lcaptcha.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbcans
            // 
            tbcans.Anchor = AnchorStyles.None;
            tbcans.BackColor = SystemColors.ControlLight;
            tbcans.Cursor = Cursors.IBeam;
            tbcans.Font = new Font("Calibri", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            tbcans.Location = new Point(235, 156);
            tbcans.MaxLength = 3;
            tbcans.Name = "tbcans";
            tbcans.PlaceholderText = "?";
            tbcans.ScrollBars = ScrollBars.Horizontal;
            tbcans.Size = new Size(40, 26);
            tbcans.TabIndex = 78;
            // 
            // lcpequal
            // 
            lcpequal.Anchor = AnchorStyles.None;
            lcpequal.AutoSize = true;
            lcpequal.BackColor = Color.Black;
            lcpequal.BorderStyle = BorderStyle.Fixed3D;
            lcpequal.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpequal.ForeColor = Color.WhiteSmoke;
            lcpequal.ImageAlign = ContentAlignment.MiddleLeft;
            lcpequal.Location = new Point(206, 158);
            lcpequal.Name = "lcpequal";
            lcpequal.Size = new Size(23, 23);
            lcpequal.TabIndex = 77;
            lcpequal.Text = "=";
            lcpequal.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp2
            // 
            lcp2.Anchor = AnchorStyles.None;
            lcp2.AutoSize = true;
            lcp2.BackColor = Color.Black;
            lcp2.BorderStyle = BorderStyle.Fixed3D;
            lcp2.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp2.ForeColor = Color.WhiteSmoke;
            lcp2.ImageAlign = ContentAlignment.MiddleLeft;
            lcp2.Location = new Point(166, 158);
            lcp2.Name = "lcp2";
            lcp2.Size = new Size(2, 23);
            lcp2.TabIndex = 76;
            lcp2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcpplus
            // 
            lcpplus.Anchor = AnchorStyles.None;
            lcpplus.AutoSize = true;
            lcpplus.BackColor = Color.Black;
            lcpplus.BorderStyle = BorderStyle.Fixed3D;
            lcpplus.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpplus.ForeColor = Color.WhiteSmoke;
            lcpplus.ImageAlign = ContentAlignment.MiddleLeft;
            lcpplus.Location = new Point(138, 158);
            lcpplus.Name = "lcpplus";
            lcpplus.Size = new Size(23, 23);
            lcpplus.TabIndex = 75;
            lcpplus.Text = "+";
            lcpplus.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp1
            // 
            lcp1.Anchor = AnchorStyles.None;
            lcp1.AutoSize = true;
            lcp1.BackColor = Color.Black;
            lcp1.BorderStyle = BorderStyle.Fixed3D;
            lcp1.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp1.ForeColor = Color.WhiteSmoke;
            lcp1.ImageAlign = ContentAlignment.MiddleLeft;
            lcp1.Location = new Point(100, 158);
            lcp1.Name = "lcp1";
            lcp1.Size = new Size(2, 23);
            lcp1.TabIndex = 74;
            lcp1.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.BackColor = Color.DarkGray;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bexit.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bexit.FlatAppearance.MouseOverBackColor = Color.Red;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.ForeColor = Color.White;
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(39, 276);
            bexit.Name = "bexit";
            bexit.Size = new Size(245, 34);
            bexit.TabIndex = 73;
            bexit.Text = "   Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.BackColor = Color.DarkGray;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bprevious.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bprevious.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.ForeColor = Color.White;
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(166, 235);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(119, 34);
            bprevious.TabIndex = 72;
            bprevious.Text = "   Previous";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // blogin
            // 
            blogin.Anchor = AnchorStyles.None;
            blogin.BackColor = Color.DarkGray;
            blogin.BackgroundImageLayout = ImageLayout.Stretch;
            blogin.Cursor = Cursors.Hand;
            blogin.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            blogin.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            blogin.FlatAppearance.MouseOverBackColor = Color.Lime;
            blogin.FlatStyle = FlatStyle.Flat;
            blogin.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            blogin.ForeColor = Color.White;
            blogin.Image = Properties.Resources.login;
            blogin.ImageAlign = ContentAlignment.MiddleLeft;
            blogin.Location = new Point(39, 235);
            blogin.Name = "blogin";
            blogin.Size = new Size(119, 34);
            blogin.TabIndex = 71;
            blogin.Text = "   Login";
            blogin.UseVisualStyleBackColor = false;
            blogin.Click += blogin_Click;
            // 
            // beyeok
            // 
            beyeok.Anchor = AnchorStyles.None;
            beyeok.BackColor = SystemColors.ScrollBar;
            beyeok.BackgroundImageLayout = ImageLayout.Stretch;
            beyeok.Cursor = Cursors.Hand;
            beyeok.FlatStyle = FlatStyle.Popup;
            beyeok.Image = (Image)resources.GetObject("beyeok.Image");
            beyeok.Location = new Point(275, 116);
            beyeok.Name = "beyeok";
            beyeok.Size = new Size(25, 23);
            beyeok.TabIndex = 70;
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
            beyecross.Location = new Point(275, 116);
            beyecross.Name = "beyecross";
            beyecross.Size = new Size(25, 23);
            beyecross.TabIndex = 69;
            beyecross.UseMnemonic = false;
            beyecross.UseVisualStyleBackColor = false;
            beyecross.Visible = false;
            beyecross.Click += beyecross_Click;
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ScrollBar;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(101, 116);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Your Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(174, 23);
            tbpass.TabIndex = 68;
            tbpass.UseSystemPasswordChar = true;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ScrollBar;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(101, 79);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Your Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 67;
            // 
            // lpass
            // 
            lpass.Anchor = AnchorStyles.None;
            lpass.AutoSize = true;
            lpass.BackColor = Color.Transparent;
            lpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lpass.ForeColor = Color.Gainsboro;
            lpass.Image = (Image)resources.GetObject("lpass.Image");
            lpass.ImageAlign = ContentAlignment.MiddleLeft;
            lpass.Location = new Point(5, 117);
            lpass.Name = "lpass";
            lpass.Size = new Size(90, 20);
            lpass.TabIndex = 66;
            lpass.Text = "     Password";
            lpass.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lusername.ForeColor = Color.Gainsboro;
            lusername.Image = Properties.Resources.username1;
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(-3, 79);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 65;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // brecapta
            // 
            brecapta.Anchor = AnchorStyles.None;
            brecapta.BackColor = SystemColors.ScrollBar;
            brecapta.BackgroundImageLayout = ImageLayout.Stretch;
            brecapta.Cursor = Cursors.Hand;
            brecapta.FlatStyle = FlatStyle.Popup;
            brecapta.Image = (Image)resources.GetObject("brecapta.Image");
            brecapta.Location = new Point(275, 156);
            brecapta.Name = "brecapta";
            brecapta.Size = new Size(25, 26);
            brecapta.TabIndex = 81;
            brecapta.UseMnemonic = false;
            brecapta.UseVisualStyleBackColor = false;
            brecapta.Click += brecapta_Click;
            // 
            // AdminLogin
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(784, 511);
            Controls.Add(panel1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "AdminLogin";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Admin Login";
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
        }

        #endregion
        private Panel panel1;
        private Label lcaptcha;
        private TextBox tbcans;
        private Label lcpequal;
        private Label lcp2;
        private Label lcpplus;
        private Label lcp1;
        private Button bexit;
        private Button bprevious;
        private Button blogin;
        private Button beyeok;
        private Button beyecross;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label lpass;
        private Label lusername;
        private PictureBox pictureBox1;
        private Button brecapta;
    }
}