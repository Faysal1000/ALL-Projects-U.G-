namespace Hungry_Heroes
{
    partial class PaymentSuccess
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(PaymentSuccess));
            label1 = new Label();
            pictureBox1 = new PictureBox();
            lamount = new Label();
            bexit = new Button();
            bhome = new Button();
            brate = new Button();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.Font = new Font("Cooper Black", 15.75F, FontStyle.Regular, GraphicsUnit.Point);
            label1.Location = new Point(195, 165);
            label1.Name = "label1";
            label1.Size = new Size(401, 24);
            label1.TabIndex = 0;
            label1.Text = "Transaction Completed Successfully.";
            // 
            // pictureBox1
            // 
            pictureBox1.Anchor = AnchorStyles.None;
            pictureBox1.Image = Properties.Resources.success;
            pictureBox1.Location = new Point(328, 12);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(154, 150);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 1;
            pictureBox1.TabStop = false;
            // 
            // lamount
            // 
            lamount.Anchor = AnchorStyles.None;
            lamount.AutoSize = true;
            lamount.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point);
            lamount.Location = new Point(295, 239);
            lamount.Name = "lamount";
            lamount.Size = new Size(118, 18);
            lamount.TabIndex = 2;
            lamount.Text = "Amount Paid: ";
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.BackColor = Color.Transparent;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bexit.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bexit.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.ForeColor = Color.Black;
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(295, 375);
            bexit.Name = "bexit";
            bexit.Size = new Size(244, 34);
            bexit.TabIndex = 81;
            bexit.Text = "Exit Program";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // bhome
            // 
            bhome.Anchor = AnchorStyles.None;
            bhome.BackColor = Color.Transparent;
            bhome.BackgroundImageLayout = ImageLayout.Stretch;
            bhome.Cursor = Cursors.Hand;
            bhome.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bhome.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bhome.FlatAppearance.MouseOverBackColor = Color.Lime;
            bhome.FlatStyle = FlatStyle.Flat;
            bhome.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bhome.ForeColor = Color.Black;
            bhome.Image = (Image)resources.GetObject("bhome.Image");
            bhome.ImageAlign = ContentAlignment.MiddleLeft;
            bhome.Location = new Point(295, 335);
            bhome.Name = "bhome";
            bhome.Size = new Size(243, 34);
            bhome.TabIndex = 80;
            bhome.Text = "Home";
            bhome.UseVisualStyleBackColor = false;
            bhome.Click += bhome_Click;
            // 
            // brate
            // 
            brate.Anchor = AnchorStyles.None;
            brate.BackColor = Color.Transparent;
            brate.BackgroundImageLayout = ImageLayout.Stretch;
            brate.Cursor = Cursors.Hand;
            brate.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            brate.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            brate.FlatAppearance.MouseOverBackColor = Color.Lime;
            brate.FlatStyle = FlatStyle.Flat;
            brate.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            brate.ForeColor = Color.Black;
            brate.Image = (Image)resources.GetObject("brate.Image");
            brate.ImageAlign = ContentAlignment.MiddleLeft;
            brate.Location = new Point(295, 295);
            brate.Name = "brate";
            brate.Size = new Size(243, 34);
            brate.TabIndex = 82;
            brate.Text = "Rate Resturant";
            brate.UseVisualStyleBackColor = false;
            brate.Click += brate_Click;
            // 
            // PaymentSuccess
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(784, 461);
            Controls.Add(brate);
            Controls.Add(bexit);
            Controls.Add(bhome);
            Controls.Add(lamount);
            Controls.Add(pictureBox1);
            Controls.Add(label1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "PaymentSuccess";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "PaymentSuccess";
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label label1;
        private PictureBox pictureBox1;
        private Label lamount;
        private Button bexit;
        private Button bhome;
        private Button brate;
    }
}